/**
 * Sidecar Supervisor — Voice Brainstorm Pad Iteration 2
 *
 * Responsibilities:
 *  - Discover local Python environment (.venv or system)
 *  - Generate cryptographically secure auth token
 *  - Spawn and supervise Python sidecar child process
 *  - Parse dynamically assigned port from sidecar stdout (VBP_SIDECAR_PORT=<port>)
 *  - Establish WebSocket connection and perform authenticated hello handshake
 *  - Health-check sidecar via periodic pings
 *  - Automatically restart sidecar on crash with exponential backoff
 *  - Expose typed state and events for Electron main and renderer
 */

import { spawn, ChildProcess } from "child_process";
import * as crypto from "crypto";
import { EventEmitter } from "events";
import * as fs from "fs";
import * as path from "path";
import WebSocket from "ws";

export type SidecarState =
  | "disconnected"
  | "connecting"
  | "ready"
  | "idle"
  | "listening"
  | "catching_up"
  | "reconnecting"
  | "error";

export interface AudioDevice {
  id: string;
  name: string;
  channels: number;
  default_samplerate: number;
}

export interface ReadyPayload {
  type: "ready";
  version: number;
  engine: string;
  model: string;
  devices: AudioDevice[];
}

export interface SupervisorOptions {
  autoStart?: boolean;
  pythonPath?: string;
  sidecarScript?: string;
  pingIntervalMs?: number;
  maxRestartDelayMs?: number;
}

export class SidecarSupervisor extends EventEmitter {
  private token: string;
  private pythonPath: string;
  private sidecarScript: string;
  private child: ChildProcess | null = null;
  private ws: WebSocket | null = null;
  private port: number = 0;
  private state: SidecarState = "disconnected";
  private isStopping: boolean = false;
  private restartTimer: NodeJS.Timeout | null = null;
  private pingTimer: NodeJS.Timeout | null = null;
  private pingTimeoutTimer: NodeJS.Timeout | null = null;
  private restartDelayMs: number = 1000;
  private readonly maxRestartDelayMs: number;
  private readonly pingIntervalMs: number;
  private devices: AudioDevice[] = [];
  private model: string = "small.en";

  constructor(options: SupervisorOptions = {}) {
    super();
    this.token = crypto.randomBytes(32).toString("hex");
    this.pythonPath = options.pythonPath ?? this.resolvePythonPath();
    this.sidecarScript =
      options.sidecarScript ??
      path.resolve(__dirname, "../../sidecar/server.py");
    this.pingIntervalMs = options.pingIntervalMs ?? 5000;
    this.maxRestartDelayMs = options.maxRestartDelayMs ?? 10000;

    if (options.autoStart) {
      this.start().catch((err) => {
        this.emit("error", err);
      });
    }
  }

  // ── Public Accessors ────────────────────────────────────────────────────────

  public getState(): SidecarState {
    return this.state;
  }

  public getPort(): number {
    return this.port;
  }

  public getToken(): string {
    return this.token;
  }

  public getPythonPath(): string {
    return this.pythonPath;
  }

  public getDevices(): AudioDevice[] {
    return this.devices;
  }

  public getModel(): string {
    return this.model;
  }

  // ── Python Discovery ────────────────────────────────────────────────────────

  private resolvePythonPath(): string {
    const candidates = [
      // .venv in Voice Brainstorm Pad root
      path.resolve(__dirname, "../../.venv/Scripts/python.exe"),
      path.resolve(__dirname, "../../.venv/bin/python"),
      // .venv in bench root (parent dir)
      path.resolve(__dirname, "../../../bench/.venv/Scripts/python.exe"),
      path.resolve(__dirname, "../../../bench/.venv/bin/python"),
    ];

    for (const cand of candidates) {
      if (fs.existsSync(cand)) {
        return cand;
      }
    }

    // System default
    return process.platform === "win32" ? "python.exe" : "python3";
  }

  // ── Lifecycle ───────────────────────────────────────────────────────────────

  public async start(): Promise<void> {
    if (this.child) return;
    this.isStopping = false;
    this.setState("connecting");

    const env = {
      ...process.env,
      PYTHONUNBUFFERED: "1",
      VBP_SIDECAR_TOKEN: this.token,
    };

    const pyArgs = [this.sidecarScript, "--token", this.token];

    return new Promise<void>((resolve, reject) => {
      let resolved = false;
      const timer = setTimeout(() => {
        if (!resolved) {
          resolved = true;
          this.setState("error");
          reject(new Error("Timed out waiting for sidecar port announcement"));
        }
      }, 8000);

      try {
        this.child = spawn(this.pythonPath, pyArgs, {
          env,
          stdio: ["ignore", "pipe", "pipe"],
        });
      } catch (err) {
        clearTimeout(timer);
        this.setState("error");
        this.scheduleRestart();
        reject(err);
        return;
      }

      // Monitor stdout for port announcement
      this.child.stdout?.on("data", (chunk: Buffer) => {
        const text = chunk.toString("utf8");
        const match = text.match(/VBP_SIDECAR_PORT=(\d+)/);
        if (match) {
          this.port = parseInt(match[1], 10);
          this.connectWebSocket(this.port);
          if (!resolved) {
            resolved = true;
            clearTimeout(timer);
            resolve();
          }
        }
      });

      // Monitor stderr for logging
      this.child.stderr?.on("data", (chunk: Buffer) => {
        const text = chunk.toString("utf8").trim();
        if (text) {
          this.emit("log", text);
        }
      });

      // Handle child exit
      this.child.on("exit", (code, signal) => {
        this.cleanupChild();
        if (!resolved) {
          resolved = true;
          clearTimeout(timer);
          reject(new Error(`Sidecar exited prematurely with code ${code}`));
        }
        if (!this.isStopping) {
          this.setState("reconnecting");
          this.scheduleRestart();
        } else {
          this.setState("disconnected");
        }
      });

      this.child.on("error", (err) => {
        this.emit("error", err);
        if (!resolved) {
          resolved = true;
          clearTimeout(timer);
          reject(err);
        }
        if (!this.isStopping) {
          this.setState("error");
          this.scheduleRestart();
        }
      });
    });
  }

  public async stop(): Promise<void> {
    this.isStopping = true;
    if (this.restartTimer) {
      clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }
    this.stopPing();

    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        // ignore
      }
      this.ws = null;
    }

    if (this.child) {
      const child = this.child;
      this.child = null;
      try {
        child.kill();
      } catch {
        // ignore
      }
    }

    this.setState("disconnected");
  }

  // ── WebSocket Connection & Handshake ────────────────────────────────────────

  private connectWebSocket(port: number): void {
    if (this.isStopping) return;

    try {
      const wsUrl = `ws://127.0.0.1:${port}`;
      const socket = new WebSocket(wsUrl);
      this.ws = socket;

      socket.on("open", () => {
        // Send initial authentication handshake (hello)
        const helloMsg = {
          type: "hello",
          version: 1,
          token: this.token,
          client: "electron-main",
        };
        socket.send(JSON.stringify(helloMsg));
        this.startPing();
      });

      socket.on("message", (raw: any) => {
        try {
          const text = typeof raw === "string" ? raw : raw.toString("utf8");
          const data = JSON.parse(text);
          this.handleMessage(data);
        } catch {
          // ignore malformed
        }
      });

      socket.on("close", () => {
        this.stopPing();
        this.ws = null;
        if (!this.isStopping && this.state !== "reconnecting") {
          this.setState("reconnecting");
          this.scheduleRestart();
        }
      });

      socket.on("error", (err: any) => {
        this.emit("error", err);
      });
    } catch (err) {
      this.scheduleRestart();
    }
  }

  private handleMessage(data: any): void {
    if (!data || typeof data !== "object") return;

    switch (data.type) {
      case "ready":
        this.model = data.model || "small.en";
        this.devices = Array.isArray(data.devices) ? data.devices : [];
        this.setState("ready");
        this.emit("ready", data);
        break;

      case "state":
        if (data.state) {
          this.setState(data.state as SidecarState);
        }
        break;

      case "level":
        this.emit("level", typeof data.rms === "number" ? data.rms : 0);
        break;

      case "chunk_ready":
        this.emit("chunk", data);
        break;

      case "pong":
        if (this.pingTimeoutTimer) {
          clearTimeout(this.pingTimeoutTimer);
          this.pingTimeoutTimer = null;
        }
        break;

      default:
        this.emit("message", data);
        break;
    }
  }

  // ── Dictation Commands (Iteration 3) ────────────────────────────────────────

  public startListening(sessionFolder: string, deviceId?: string): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(
        JSON.stringify({
          type: "start",
          version: 1,
          session_folder: sessionFolder,
          device_id: deviceId || null,
        })
      );
    }
  }

  public stopListening(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "stop", version: 1 }));
    }
  }

  public setDevice(deviceId: string): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(
        JSON.stringify({
          type: "set_device",
          version: 1,
          device_id: deviceId,
        })
      );
    }
  }

  // ── Ping Health Check ───────────────────────────────────────────────────────

  private startPing(): void {
    this.stopPing();
    this.pingTimer = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: "ping", version: 1, id: Date.now() }));
        // Expect pong within 3s
        this.pingTimeoutTimer = setTimeout(() => {
          // Pong timed out — connection may be hung
          if (this.ws) {
            try {
              this.ws.close();
            } catch {
              // ignore
            }
          }
        }, 3000);
      }
    }, this.pingIntervalMs);
  }

  private stopPing(): void {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
    if (this.pingTimeoutTimer) {
      clearTimeout(this.pingTimeoutTimer);
      this.pingTimeoutTimer = null;
    }
  }

  // ── Helpers ─────────────────────────────────────────────────────────────────

  private setState(newState: SidecarState): void {
    if (this.state !== newState) {
      this.state = newState;
      this.emit("stateChange", newState);
    }
  }

  private scheduleRestart(): void {
    if (this.isStopping || this.restartTimer) return;

    this.restartTimer = setTimeout(() => {
      this.restartTimer = null;
      this.restartDelayMs = Math.min(
        this.restartDelayMs * 1.5,
        this.maxRestartDelayMs
      );
      this.start().catch(() => {});
    }, this.restartDelayMs);
  }

  private cleanupChild(): void {
    if (this.child) {
      this.child = null;
    }
    this.port = 0;
    this.stopPing();
  }
}
