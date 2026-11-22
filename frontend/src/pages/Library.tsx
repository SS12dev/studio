import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Job, type PhotoItem } from "../api";
import PhotoViewer from "../components/PhotoViewer";

const PAGE = 120;

export default function Library({ onChange }: { onChange: () => void }) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [photos, setPhotos] = useState<PhotoItem[]>([]);
  const [total, setTotal] = useState(0);
  const [path, setPath] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [open, setOpen] = useState<number | null>(null);
  const dragDepth = useRef(0);

  const loadPhotos = useCallback(async (reset = true) => {
    const r = await api.photos(reset ? 0 : photos.length, PAGE);
    setTotal(r.total);
    setPhotos((prev) => (reset ? r.items : [...prev, ...r.items]));
  }, [photos.length]);

  const running = jobs.some((j) => j.status === "queued" || j.status === "running");

  useEffect(() => { api.imports().then(setJobs); loadPhotos(true); }, []); // eslint-disable-line
  useEffect(() => {
    if (!running) return;
    const t = setInterval(async () => {
      const js = await api.imports();
      setJobs(js);
      await loadPhotos(true);
      onChange();
    }, 1000);
    return () => clearInterval(t);
  }, [running, loadPhotos, onChange]);

  const start = async (p: Promise<{ job_id: number }>) => {
    setError(null);
    try { await p; setJobs(await api.imports()); } catch (e) { setError(String((e as Error).message)); }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); dragDepth.current = 0; setDragging(false);
    const files = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith("image/") ||
      /\.(heic|heif|dng|cr2|cr3|nef|arw|raf|rw2|orf|tif|tiff)$/i.test(f.name));
    if (files.length) start(api.importFiles(files));
    else setError("Those don't look like photos. Folders can't be dropped; use the path box instead.");
  };

  return (
    <main className="library"
      onDragEnter={(e) => { e.preventDefault(); dragDepth.current++; setDragging(true); }}
      onDragLeave={() => { if (--dragDepth.current <= 0) setDragging(false); }}
      onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
      <section className="import">
        <form onSubmit={(e) => { e.preventDefault(); if (path.trim()) start(api.importFolder(path.trim())); }}>
          <input value={path} onChange={(e) => setPath(e.target.value)}
            placeholder="Folder on this Mac, e.g. ~/Pictures/friends" />
          <button className="primary" disabled={!path.trim()}>Import folder</button>
        </form>
        <span className="muted">or drop photos anywhere on this page. Folders are imported in place;
          dropped files are copied into the library.</span>
        {error && <p className="error">{error}</p>}
        {jobs.slice(0, 4).map((j) => <JobRow key={j.id} job={j} />)}
      </section>

      {photos.length === 0 ? (
        <div className="empty">No photos yet. Import a folder to start.</div>
      ) : (
        <>
          <div className="muted count">{total} photos</div>
          <div className="photos">
            {photos.map((p) => (
              <figure key={p.id} onClick={() => setOpen(p.id)} title={p.filename}>
                <img src={p.thumb} loading="lazy" alt={p.filename} />
                {p.face_count > 0 && <span className="badge">{p.face_count}</span>}
              </figure>
            ))}
          </div>
          {photos.length < total && (
            <button className="more" onClick={() => loadPhotos(false)}>Load more</button>
          )}
        </>
      )}
      {dragging && <div className="dropzone">Drop photos to import</div>}
      {open !== null && <PhotoViewer photoId={open} onClose={() => setOpen(null)} onChange={onChange} />}
    </main>
  );
}

function JobRow({ job }: { job: Job }) {
  const pct = job.total ? Math.round((100 * job.processed) / job.total) : 0;
  const active = job.status === "running" || job.status === "queued";
  const [why, setWhy] = useState<Awaited<ReturnType<typeof api.importFailures>> | null>(null);
  return (
    <div className={`job ${job.status}`}>
      <div className="jobline">
        <strong>{job.source}</strong>
        <span className="muted">
          {active ? `${job.processed} / ${job.total}` : job.status}
          {" · "}{job.added} new, {job.duplicates} already in library, {job.faces} faces
          {job.auto_filed ? `, ${job.auto_filed} filed to people` : ""}
          {job.failed ? `, ${job.failed} failed` : ""}
        </span>
        {job.failed > 0 && <button className="link" onClick={async () =>
          setWhy(why ? null : await api.importFailures(job.id))}>{why ? "hide" : "why?"}</button>}
        {active && <button onClick={() => api.cancelImport(job.id)}>Cancel</button>}
      </div>
      {active && <div className="bar"><div style={{ width: `${pct}%` }} /></div>}
      {job.message && job.status !== "done" && <div className="muted small">{job.message}</div>}
      {why && (
        <div className="failures small">
          {why.total === 0 && <p className="muted">No details were saved for this import (it ran before
            failure details existed). Import the folder again: photos already in the library are skipped and only
            the failed ones are retried.</p>}
          {why.groups.map((g) => (
            <details key={g.reason}>
              <summary><b>{g.count}×</b> {g.error}</summary>
              {g.examples.map((x) => <div key={x} className="muted">{x}</div>)}
            </details>
          ))}
          {why.total > 0 && <p className="muted">Import the folder again after a fix: only these files are retried.</p>}
        </div>
      )}
    </div>
  );
}
