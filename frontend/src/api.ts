// Thin typed client for the Studio API.

export type Job = {
  id: number; source: string; status: "queued" | "running" | "done" | "error" | "cancelled";
  total: number; processed: number; added: number; duplicates: number; failed: number;
  faces: number; auto_filed: number; message: string | null; created_at: string;
};
export type PhotoItem = {
  id: number; filename: string; thumb: string; width: number; height: number;
  face_count: number; taken_at: string | null;
};
export type FaceItem = {
  id: number; photo_id: number; crop: string; thumb: string | null; filename: string | null;
  person_id: number | null; source: "confirmed" | "auto" | "review" | null;
  guess_person_id: number | null; score: number | null; usable: boolean;
  reject_reason: string | null; second_pass: boolean; no_training: boolean;
};
export type PersonItem = {
  id: number; name: string; faces: number; confirmed: number; no_training: number;
  cover: string | null;
};
export type Group = { key: number; faces: FaceItem[] };
export type Stats = {
  photos: number; faces: number; people: number; review: number; groups: number; unsorted: number;
};
export type PhotoDetail = {
  id: number; filename: string; path: string; width: number; height: number;
  preview: string; faces: (FaceItem & { box: [number, number, number, number]; person: string | null })[];
};

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = `${r.status}`;
    try { msg = (await r.json()).detail ?? msg; } catch { /* not json */ }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}
const post = <T,>(url: string, body?: unknown) =>
  req<T>(url, { method: "POST", headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  stats: () => req<Stats>("/api/stats"),
  imports: () => req<Job[]>("/api/imports"),
  importFolder: (path: string) => post<{ job_id: number }>("/api/imports/folder", { path }),
  importFiles: (files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f, f.name));
    return req<{ job_id: number }>("/api/imports/upload", { method: "POST", body: fd });
  },
  cancelImport: (id: number) => post(`/api/imports/${id}/cancel`),
  photos: (offset = 0, limit = 120, personId?: number) =>
    req<{ total: number; items: PhotoItem[] }>(
      `/api/photos?offset=${offset}&limit=${limit}${personId ? `&person_id=${personId}` : ""}`),
  photo: (id: number) => req<PhotoDetail>(`/api/photos/${id}`),
  people: () => req<PersonItem[]>("/api/people"),
  personFaces: (id: number) => req<FaceItem[]>(`/api/people/${id}/faces`),
  rename: (id: number, name: string) => post<{ person_id: number }>(`/api/people/${id}/rename`, { name }),
  groups: () => req<Group[]>("/api/groups"),
  findGroups: (link_sim = 0.25) => post<{ groups: number }>("/api/groups/refresh", { link_sim }),
  nameGroups: (keys: number[], name: string) =>
    post<{ person_id: number; auto_filed: number }>("/api/groups/name", { keys, name }),
  review: () => req<FaceItem[]>("/api/review"),
  unsorted: () => req<FaceItem[]>("/api/unsorted"),
  autoFile: () => post<{ auto_filed: number }>("/api/people/auto-file"),
  remove: (person_id: number, face_ids: number[]) => post("/api/faces/remove", { person_id, face_ids }),
  move: (face_ids: number[], name: string) => post<{ person_id: number }>("/api/faces/move", { face_ids, name }),
  training: (face_ids: number[], exclude: boolean) => post("/api/faces/training", { face_ids, exclude }),
  answer: (face_id: number, name: string | null) => post("/api/faces/answer", { face_id, name }),
};
