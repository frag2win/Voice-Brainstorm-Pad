/**
 * Tests for SidecarSupervisor — Voice Brainstorm Pad Iteration 2
 *
 * Tests cover:
 *  - Supervisor state initialization
 *  - Finding Python executable in .venv or system
 *  - Generating secure crypto token
 *  - Starting sidecar and completing handshake
 *  - Ping-pong health checks
 *  - Graceful stop
 */

import { describe, it, expect, afterEach } from "vitest";
import { SidecarSupervisor } from "../src/main/sidecar-supervisor";
import WebSocket, { WebSocketServer } from "ws";
import * as path from "path";

describe("SidecarSupervisor", () => {
  let supervisor: SidecarSupervisor | null = null;

  afterEach(async () => {
    if (supervisor) {
      await supervisor.stop();
      supervisor = null;
    }
  });

  it("generates a 64-character hex auth token", () => {
    supervisor = new SidecarSupervisor({ autoStart: false });
    expect(supervisor.getToken()).toMatch(/^[0-9a-f]{64}$/);
  });

  it("locates the python executable in .venv", () => {
    supervisor = new SidecarSupervisor({ autoStart: false });
    const py = supervisor.getPythonPath();
    expect(py).toBeTruthy();
    expect(py.toLowerCase()).toContain("python");
  });

  it("initial state is disconnected before start", () => {
    supervisor = new SidecarSupervisor({ autoStart: false });
    expect(supervisor.getState()).toBe("disconnected");
  });

  it("starts sidecar, receives port, completes handshake, and reaches idle state", async () => {
    supervisor = new SidecarSupervisor({ autoStart: false });

    const states: string[] = [];
    supervisor.on("stateChange", (state: string) => {
      states.push(state);
    });

    await supervisor.start();

    // Port should be assigned and > 0
    expect(supervisor.getPort()).toBeGreaterThan(0);

    // Wait until sidecar completes model warmup and transitions to idle
    const startWait = Date.now();
    while (supervisor.getState() !== "idle" && Date.now() - startWait < 8000) {
      await new Promise((r) => setTimeout(r, 100));
    }

    expect(supervisor.getState()).toBe("idle");
    expect(states).toContain("connecting");
    expect(states).toContain("ready");
    expect(states).toContain("idle");
  }, 10000);

  it("handles stop cleanly and transitions to disconnected", async () => {
    supervisor = new SidecarSupervisor({ autoStart: false });
    await supervisor.start();
    await new Promise((r) => setTimeout(r, 800));

    await supervisor.stop();
    expect(supervisor.getState()).toBe("disconnected");
  }, 10000);

  it("tolerates up to 2 missed pongs, resets counter on pong, and closes socket only on 3 consecutive missed pongs", async () => {
    const wss = new WebSocketServer({ port: 0 });
    const address = wss.address() as any;
    const port = address.port;

    let pingCount = 0;
    wss.on("connection", (ws) => {
      ws.on("message", (msg) => {
        try {
          const data = JSON.parse(msg.toString());
          if (data.type === "ping") {
            pingCount++;
            // Answer ping #3 with a pong to test reset
            if (pingCount === 3) {
              ws.send(JSON.stringify({ type: "pong", version: 1, id: data.id }));
            }
          }
        } catch {
          // ignore
        }
      });
    });

    supervisor = new SidecarSupervisor({
      autoStart: false,
      pingIntervalMs: 60,
      pingTimeoutMs: 25,
    });

    (supervisor as any).connectWebSocket(port);

    // Wait for connection and pings 1 and 2 to time out
    const waitMiss2 = Date.now();
    while (supervisor.getMissedPongs() < 2 && Date.now() - waitMiss2 < 3000) {
      await new Promise((r) => setTimeout(r, 20));
    }
    expect(supervisor.getMissedPongs()).toBe(2);
    expect((supervisor as any).ws).not.toBeNull();
    expect((supervisor as any).ws.readyState).toBe(WebSocket.OPEN);

    // Wait for ping 3 which is answered with pong and resets counter
    const waitReset = Date.now();
    while (supervisor.getMissedPongs() !== 0 && Date.now() - waitReset < 3000) {
      await new Promise((r) => setTimeout(r, 20));
    }
    expect(supervisor.getMissedPongs()).toBe(0);
    expect((supervisor as any).ws).not.toBeNull();
    expect((supervisor as any).ws.readyState).toBe(WebSocket.OPEN);

    // Wait for 3 consecutive missed pongs
    const waitClose = Date.now();
    while (supervisor.getMissedPongs() < 3 && Date.now() - waitClose < 3000) {
      await new Promise((r) => setTimeout(r, 20));
    }
    expect(supervisor.getMissedPongs()).toBeGreaterThanOrEqual(3);
    await new Promise((r) => setTimeout(r, 60));
    const ws = (supervisor as any).ws;
    const isClosedOrClosing = !ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING;
    expect(isClosedOrClosing).toBe(true);

    wss.close();
  });
});
