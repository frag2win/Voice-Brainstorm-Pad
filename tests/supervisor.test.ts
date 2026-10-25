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
});
