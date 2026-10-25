/**
 * Preload script — Voice Brainstorm Pad Iteration 1
 *
 * Exposes a narrow, typed API to the renderer via contextBridge.
 * The renderer has zero access to Node.js or Electron internals.
 */

import { contextBridge, ipcRenderer } from "electron";

export interface SessionMeta {
  slug: string;
  title: string;
  created: string;
  updated: string;
  clean_shutdown: boolean;
}

export interface SessionPayload {
  meta: SessionMeta;
  text: string;
  folderPath: string;
}

export interface SidecarStatus {
  state: string;
  model: string;
  devices: Array<{ id: string; name: string }>;
  port: number;
}

export interface VbpAPI {
  sessions: {
    list(): Promise<SessionMeta[]>;
    create(title: string): Promise<SessionPayload>;
    open(slug: string): Promise<SessionPayload | null>;
    save(slug: string, text: string): Promise<void>;
    rename(slug: string, title: string): Promise<void>;
    delete(slug: string): Promise<void>;
    rootPath(): Promise<string>;
  };
  sidecar: {
    getStatus(): Promise<SidecarStatus>;
    restart(): Promise<void>;
    startListening(sessionFolder: string, deviceId?: string): Promise<void>;
    stopListening(): Promise<void>;
    setDevice(deviceId: string): Promise<void>;
    onStateChange(callback: (state: string) => void): () => void;
    onLevel(callback: (rms: number) => void): () => void;
    onChunk(callback: (chunk: any) => void): () => void;
    onTranscript(callback: (data: any) => void): () => void;
    onChunkFailed(callback: (data: any) => void): () => void;
  };
}

const api: VbpAPI = {
  sessions: {
    list: () => ipcRenderer.invoke("sessions:list"),
    create: (title: string) => ipcRenderer.invoke("sessions:create", { title }),
    open: (slug: string) => ipcRenderer.invoke("sessions:open", { slug }),
    save: (slug: string, text: string) =>
      ipcRenderer.invoke("sessions:save", { slug, text }),
    rename: (slug: string, title: string) =>
      ipcRenderer.invoke("sessions:rename", { slug, title }),
    delete: (slug: string) => ipcRenderer.invoke("sessions:delete", { slug }),
    rootPath: () => ipcRenderer.invoke("sessions:rootPath"),
  },
  sidecar: {
    getStatus: () => ipcRenderer.invoke("sidecar:status"),
    restart: () => ipcRenderer.invoke("sidecar:restart"),
    startListening: (sessionFolder: string, deviceId?: string) =>
      ipcRenderer.invoke("sidecar:start", { sessionFolder, deviceId }),
    stopListening: () => ipcRenderer.invoke("sidecar:stop"),
    setDevice(deviceId: string) {
      return ipcRenderer.invoke("sidecar:setDevice", { deviceId });
    },
    onStateChange: (callback: (state: string) => void) => {
      const handler = (_event: any, state: string) => callback(state);
      ipcRenderer.on("sidecar:state", handler);
      return () => {
        ipcRenderer.removeListener("sidecar:state", handler);
      };
    },
    onLevel: (callback: (rms: number) => void) => {
      const handler = (_event: any, rms: number) => callback(rms);
      ipcRenderer.on("sidecar:level", handler);
      return () => {
        ipcRenderer.removeListener("sidecar:level", handler);
      };
    },
    onChunk: (callback: (chunk: any) => void) => {
      const handler = (_event: any, chunk: any) => callback(chunk);
      ipcRenderer.on("sidecar:chunk", handler);
      return () => {
        ipcRenderer.removeListener("sidecar:chunk", handler);
      };
    },
    onTranscript: (callback: (data: any) => void) => {
      const handler = (_event: any, data: any) => callback(data);
      ipcRenderer.on("sidecar:transcript", handler);
      return () => {
        ipcRenderer.removeListener("sidecar:transcript", handler);
      };
    },
    onChunkFailed: (callback: (data: any) => void) => {
      const handler = (_event: any, data: any) => callback(data);
      ipcRenderer.on("sidecar:chunk_failed", handler);
      return () => {
        ipcRenderer.removeListener("sidecar:chunk_failed", handler);
      };
    },
  },
};

contextBridge.exposeInMainWorld("vbp", api);
