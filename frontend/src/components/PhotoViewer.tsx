import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type FaceItem, type PersonItem, type PhotoDetail } from "../api";

type Box = [number, number, number, number];
type Suggest = { from: number; fromName: string; to: string; faces: (FaceItem & { sim: number })[]; sel: Set<number> };

type Props = { photoId: number; focusFaceId?: number; onClose: () => void; onChange?: () => void };

/** One photo with its faces. Click a face to name it, create a new person, mark it "not them"
 *  or "not a face". Draw a box for a face the detector missed or boxed badly. */
export default function PhotoViewer({ photoId, focusFaceId, onClose, onChange }: Props) {
  const [d, setD] = useState<PhotoDetail | null>(null);
  const [people, setPeople] = useState<PersonItem[]>([]);
  const [cur, setCur] = useState<number | null>(focusFaceId ?? null);
  const [name, setName] = useState("");
  const [drawing, setDrawing] = useState(false);
  const [drag, setDrag] = useState<Box | null>(null);
  const [suggest, setSuggest] = useState<Suggest | null>(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  const svg = useRef<SVGSVGElement>(null);
  const input = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    const [p, ppl] = await Promise.all([api.photo(photoId), api.people()]);
    setD(p); setPeople(ppl);
    return p;
  }, [photoId]);
  useEffect(() => { load(); }, [load]);
  useEffect(() => { setName(""); }, [cur]);

  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (document.activeElement === input.current) input.current?.blur();
      else if (drawing) { setDrawing(false); setDrag(null); }
      else onClose();
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose, drawing]);

  const face = d?.faces.find((f) => f.id === cur);
  const known = useMemo(() => new Set(people.map((p) => p.name.toLowerCase())), [people]);
  const twice = useMemo(() => {
    const n: Record<string, number> = {};
    d?.faces.forEach((f) => { if (f.person) n[f.person] = (n[f.person] ?? 0) + 1; });
    return Object.keys(n).filter((k) => n[k] > 1);
  }, [d]);

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true); setNote("");
    try { await fn(); await load(); onChange?.(); }
    catch (e) { setNote(`⚠ ${(e as Error).message}`); }
    finally { setBusy(false); }
  };

  const setPerson = (target: string) => face && target.trim() && run(async () => {
    const to = target.trim();
    const prev = face.person_id;
    const prevName = face.person;
    await api.move([face.id], to);
    setNote(known.has(to.toLowerCase()) ? `Set to ${to}.` : `Created ${to}.`);
    if (prev && prevName && prevName !== to) {
      const hits = await api.similar(face.id, prev);
      if (hits.length) setSuggest({ from: prev, fromName: prevName, to, faces: hits, sel: new Set(hits.map((h) => h.id)) });
    }
  });
  const notThem = () => face?.person_id && run(() => api.remove(face.person_id!, [face.id]));
  const notFace = (v: boolean) => face && run(() => api.notFace([face.id], v));
  const training = () => face && run(() => api.training([face.id], !face.no_training));
  const moveSuggested = () => suggest && run(async () => {
    await api.move([...suggest.sel], suggest.to);
    setNote(`Moved ${suggest.sel.size} more faces from ${suggest.fromName} to ${suggest.to}.`);
    setSuggest(null);
  });

  // drawing --------------------------------------------------------------------
  const toImg = (e: React.MouseEvent): [number, number] => {
    const el = svg.current!;
    const pt = el.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const p = pt.matrixTransform(el.getScreenCTM()!.inverse());
    return [Math.max(0, Math.min(d!.width, p.x)), Math.max(0, Math.min(d!.height, p.y))];
  };
  const down = (e: React.MouseEvent) => { if (drawing) { const [x, y] = toImg(e); setDrag([x, y, x, y]); } };
  const moveDrag = (e: React.MouseEvent) => { if (drawing && drag) { const [x, y] = toImg(e); setDrag([drag[0], drag[1], x, y]); } };
  const up = () => {
    if (!drawing || !drag) return;
    const b: Box = [Math.min(drag[0], drag[2]), Math.min(drag[1], drag[3]), Math.max(drag[0], drag[2]), Math.max(drag[1], drag[3])];
    setDrag(null); setDrawing(false);
    if (b[2] - b[0] < 8 || b[3] - b[1] < 8) return;
    run(async () => {
      const r = await api.addFace(photoId, b);
      setCur(r.face_id);
      setNote(r.found ? "Face found in your box. Give it a name."
        : "No face found there, so it can't be matched automatically, but you can still name it.");
    });
  };

  const fs = d ? Math.max(d.width, d.height) / 50 : 20; // label size in image pixels
  const cls = (f: PhotoDetail["faces"][number]) =>
    [f.not_face ? "notface" : f.person ? "named" : f.usable ? "unknown" : "aside",
     f.id === cur ? "cur" : "", f.person && twice.includes(f.person) ? "twice" : ""].join(" ");
  const label = (f: PhotoDetail["faces"][number]) =>
    f.not_face ? "not a face" : f.person ?? (f.source === "review" ? "needs answer" : "unknown");

  return (
    <div className="modal" onClick={onClose}>
      <div className="viewer" onClick={(e) => e.stopPropagation()}>
        {d && (
          <div className="viewerbody">
            <div className={`stage ${drawing ? "drawing" : ""}`}>
              <img src={d.preview} alt={d.filename} />
              <svg ref={svg} viewBox={`0 0 ${d.width} ${d.height}`} preserveAspectRatio="xMidYMid meet"
                onMouseDown={down} onMouseMove={moveDrag} onMouseUp={up}
                onClick={(e) => { if (!drawing && e.target === svg.current) setCur(null); }}>
                {d.faces.map((f) => (
                  <g key={f.id} onClick={(e) => { if (!drawing) { e.stopPropagation(); setCur(f.id); } }}>
                    <rect x={f.box[0]} y={f.box[1]} width={f.box[2] - f.box[0]} height={f.box[3] - f.box[1]} className={cls(f)} />
                    <text x={f.box[0]} y={f.box[1] > fs * 1.2 ? f.box[1] - fs * 0.3 : f.box[3] + fs}
                      fontSize={fs} className={f.not_face ? "dim" : ""}>{label(f)}</text>
                  </g>
                ))}
                {drag && <rect className="draft" x={Math.min(drag[0], drag[2])} y={Math.min(drag[1], drag[3])}
                  width={Math.abs(drag[2] - drag[0])} height={Math.abs(drag[3] - drag[1])} />}
              </svg>
            </div>

            <aside className="facepanel">
              <div className="row">
                <strong className="grow ellipsis">{d.filename}</strong>
                <button onClick={onClose}>Close</button>
              </div>
              <p className="muted small ellipsis" title={d.path}>{d.width}×{d.height} · {d.path}</p>
              {twice.length > 0 && <div className="warnbox">{twice.join(", ")} is on more than one face here. Click the wrong one and fix it.</div>}

              <div className="facelist">
                {d.faces.map((f) => (
                  <button key={f.id} className={`chipface ${f.id === cur ? "on" : ""} ${f.not_face ? "dim" : ""}`} onClick={() => setCur(f.id)}>
                    <img src={f.crop} alt="" /><span>{label(f)}</span>
                  </button>
                ))}
              </div>
              <button className={drawing ? "primary" : ""} onClick={() => { setDrawing(!drawing); setDrag(null); }}>
                {drawing ? "Drag over the face… (Esc to cancel)" : "+ Draw a face box"}</button>

              {face && (
                <div className="faceedit">
                  <div className="row">
                    <img className="big" src={face.crop} alt="" />
                    <div className="small">
                      <div><b>{label(face)}</b></div>
                      <div className="muted">
                        {face.source === "auto" ? "filed by the matcher" : face.source === "confirmed" ? "set by you" :
                          face.source === "review" ? "matcher unsure" : ""}
                        {face.manual ? " · drawn by you" : ""}
                        {!face.usable && face.reject_reason ? ` · ${face.reject_reason.replace("_", " ")}` : ""}
                        {face.no_training ? " · no training" : ""}
                      </div>
                    </div>
                  </div>
                  <form className="row" onSubmit={(e) => { e.preventDefault(); setPerson(name); }}>
                    <input ref={input} className="grow" list="viewer-names" value={name} autoFocus
                      onChange={(e) => setName(e.target.value)} placeholder="Who is this? Type a name" />
                    <datalist id="viewer-names">{people.map((p) => <option key={p.id} value={p.name} />)}</datalist>
                    <button className="primary" disabled={busy || !name.trim()}>
                      {name.trim() && !known.has(name.trim().toLowerCase()) ? "Create" : "Set"}</button>
                  </form>
                  <div className="chips">
                    {people.filter((p) => p.name !== face.person).slice(0, 10).map((p) => (
                      <button key={p.id} disabled={busy} onClick={() => setPerson(p.name)}>{p.name}</button>
                    ))}
                  </div>
                  <div className="row wrap">
                    {face.person && <button className="warn" disabled={busy} onClick={notThem}>Not {face.person}</button>}
                    {face.not_face
                      ? <button disabled={busy} onClick={() => notFace(false)}>It is a face</button>
                      : <button className="warn" disabled={busy} onClick={() => notFace(true)}>Not a face</button>}
                    {!face.not_face && <button disabled={busy} onClick={training}>
                      {face.no_training ? "Use for training" : "Exclude from training"}</button>}
                  </div>
                </div>
              )}
              {!face && !drawing && <p className="muted small">Click a face to name it or fix it. If a box is wrong
                (not a face, or only part of one), mark it "Not a face" and draw the right box.</p>}

              {suggest && (
                <div className="suggest">
                  <p><b>{suggest.faces.length}</b> more {suggest.faces.length === 1 ? "face" : "faces"} in {suggest.fromName} look
                    like this one. Move the selected ones to {suggest.to}?</p>
                  <div className="minigrid">
                    {suggest.faces.map((f) => (
                      <img key={f.id} src={f.crop} alt="" title={`similarity ${f.sim}`}
                        className={suggest.sel.has(f.id) ? "on" : ""}
                        onClick={() => {
                          const s = new Set(suggest.sel);
                          if (s.has(f.id)) s.delete(f.id); else s.add(f.id);
                          setSuggest({ ...suggest, sel: s });
                        }} />
                    ))}
                  </div>
                  <div className="row">
                    <button className="primary" disabled={busy || !suggest.sel.size} onClick={moveSuggested}>
                      Move {suggest.sel.size}</button>
                    <button onClick={() => setSuggest(null)}>Not now</button>
                  </div>
                </div>
              )}
              {note && <p className="note small">{note}</p>}
            </aside>
          </div>
        )}
      </div>
    </div>
  );
}
