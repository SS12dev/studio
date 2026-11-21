import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type Conflict, type FaceItem, type Group, type PersonItem } from "../api";
import FaceGrid from "../components/FaceGrid";
import PhotoViewer from "../components/PhotoViewer";

type View =
  | { kind: "review" }
  | { kind: "person"; id: number }
  | { kind: "group"; key: number }
  | { kind: "unsorted" }
  | { kind: "conflicts" };

export default function People({ onChange }: { onChange: () => void }) {
  const [people, setPeople] = useState<PersonItem[]>([]);
  const [groups, setGroups] = useState<Group[]>([]);
  const [review, setReview] = useState<FaceItem[]>([]);
  const [unsorted, setUnsorted] = useState<FaceItem[]>([]);
  const [faces, setFaces] = useState<FaceItem[]>([]);
  const [view, setView] = useState<View>({ kind: "review" });
  const [sel, setSel] = useState<Set<number>>(new Set());
  const [hover, setHover] = useState<FaceItem | null>(null);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  const [conflicts, setConflicts] = useState<Conflict[]>([]);
  const [open, setOpen] = useState<FaceItem | null>(null);
  const [armDelete, setArmDelete] = useState(false);

  const names = useMemo(() => Object.fromEntries(people.map((p) => [p.id, p.name])), [people]);

  const refresh = useCallback(async () => {
    const [p, g, r, c] = await Promise.all([api.people(), api.groups(), api.review(), api.conflicts()]);
    setPeople(p); setGroups(g); setReview(r); setConflicts(c);
    onChange();
    return { p, g, r };
  }, [onChange]);

  const loadView = useCallback(async (v: View) => {
    if (v.kind === "person") setFaces(await api.personFaces(v.id));
    else if (v.kind === "unsorted") { const u = await api.unsorted(); setUnsorted(u); setFaces(u); }
  }, []);

  useEffect(() => {
    refresh().then(({ p, r }) => { if (!r.length && p.length) setView({ kind: "person", id: p[0].id }); });
  }, []); // eslint-disable-line
  useEffect(() => { setSel(new Set()); setArmDelete(false); loadView(view); }, [view, loadView]);

  const shown: FaceItem[] =
    view.kind === "review" ? review :
    view.kind === "group" ? groups.find((g) => g.key === view.key)?.faces ?? [] :
    view.kind === "conflicts" ? conflicts.flatMap((c) => c.faces) : faces;

  const run = async (fn: () => Promise<unknown>, msg?: string) => {
    setBusy(true); setNote("");
    try { await fn(); setSel(new Set()); await refresh(); await loadView(view); if (msg) setNote(msg); }
    catch (e) { setNote(`⚠ ${(e as Error).message}`); }
    finally { setBusy(false); }
  };
  const ids = () => [...sel];
  const need = (what: string) => { setNote(`⚠ ${what}`); return false; };

  const act = {
    notThem: () => (sel.size || need("select faces first")) && run(async () => {
      const by: Record<number, number[]> = {};
      for (const f of shown.filter((x) => sel.has(x.id) && x.person_id)) (by[f.person_id!] ??= []).push(f.id);
      for (const [pid, fids] of Object.entries(by)) await api.remove(Number(pid), fids);
    }),
    notFace: () => (sel.size || need("select faces first")) &&
      run(() => api.notFace(ids()), `${sel.size} marked as not a face. They're hidden everywhere now.`),
    deletePerson: () => {
      if (view.kind !== "person") return;
      if (!armDelete) { setArmDelete(true); setNote("Click again to delete. Their faces go back to Unsorted."); return; }
      run(async () => { await api.deletePerson(view.id); setArmDelete(false); setView({ kind: "review" }); });
    },
    training: (exclude: boolean) => (sel.size || need("select faces first")) &&
      run(() => api.training(ids(), exclude)),
    moveTo: () => (sel.size || need("select faces first")) && (name.trim() || need("type a name")) &&
      run(() => api.move(ids(), name.trim())),
    accept: () => (sel.size || need("select faces first")) && run(async () => {
      for (const f of review.filter((x) => sel.has(x.id)))
        await api.answer(f.id, f.guess_person_id ? names[f.guess_person_id] : null);
    }),
    reject: () => (sel.size || need("select faces first")) && run(async () => {
      for (const id of ids()) await api.answer(id, null);
    }),
    nameGroup: () => view.kind === "group" && (name.trim() || need("type a name")) &&
      run(async () => {
        const g = groups.find((x) => x.key === view.key)!;
        if (sel.size && sel.size < g.faces.length) await api.move(ids(), name.trim());
        else {
          const r = await api.nameGroups([view.key], name.trim());
          setView({ kind: "person", id: r.person_id });
          setNote(`Named. ${r.auto_filed} more faces were filed automatically.`);
        }
      }),
    rename: () => view.kind === "person" && (name.trim() || need("type the new name")) &&
      run(async () => { const r = await api.rename(view.id, name.trim()); setView({ kind: "person", id: r.person_id }); }),
    findGroups: () => run(async () => { const r = await api.findGroups(); setNote(`${r.groups} groups found`); }),
    autoFile: () => run(async () => { const r = await api.autoFile(); setNote(`${r.auto_filed} faces filed`); }),
  };

  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT" || open) return;
      const key = e.key.toLowerCase();
      if (key === "escape") setSel(new Set());
      else if (key === "a" && !e.metaKey) setSel(new Set(shown.map((f) => f.id)));
      else if (key === "r") view.kind === "review" ? act.reject() : act.notThem();
      else if (key === "f") act.notFace();
      else if (key === "t") act.training(!shown.filter((f) => sel.has(f.id)).every((f) => f.no_training));
      else if (key === "y" && view.kind === "review") act.accept();
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  });

  const person = view.kind === "person" ? people.find((p) => p.id === view.id) : undefined;
  const title = view.kind === "review" ? "Needs your answer" : view.kind === "unsorted" ? "Unsorted faces" :
    view.kind === "conflicts" ? "Same person twice in a photo" :
    view.kind === "group" ? `Unnamed group` : person?.name ?? "";

  return (
    <main className="people">
      <nav className="side">
        <h4>Review</h4>
        <a className={view.kind === "review" ? "on" : ""} onClick={() => setView({ kind: "review" })}>
          Needs your answer <small>{review.length}</small></a>
        <a className={view.kind === "conflicts" ? "on" : ""} onClick={() => setView({ kind: "conflicts" })}>
          Same person twice <small>{conflicts.length || ""}</small></a>
        <h4>People</h4>
        {people.map((p) => (
          <a key={p.id} className={view.kind === "person" && view.id === p.id ? "on" : ""}
            onClick={() => setView({ kind: "person", id: p.id })}>
            {p.cover ? <img src={p.cover} alt="" /> : <span className="noimg" />}
            <span className="grow">{p.name}</span><small>{p.faces}</small></a>
        ))}
        {!people.length && <p className="muted small">Name a group below to create a person.</p>}
        <h4>Unnamed groups <button className="link" disabled={busy} onClick={act.findGroups}>find</button></h4>
        {groups.map((g, i) => (
          <a key={g.key} className={view.kind === "group" && view.key === g.key ? "on" : ""}
            onClick={() => setView({ kind: "group", key: g.key })}>
            <img src={g.faces[0].crop} alt="" /><span className="grow">Group {i + 1}</span>
            <small>{g.faces.length}</small></a>
        ))}
        {!groups.length && <p className="muted small">Click "find" to group faces that have no name yet.</p>}
        <h4>Other</h4>
        <a className={view.kind === "unsorted" ? "on" : ""} onClick={() => setView({ kind: "unsorted" })}>
          Unsorted faces <small>{view.kind === "unsorted" ? unsorted.length : ""}</small></a>
      </nav>

      <section className="main">
        <div className="toolbar">
          <h2>{title}</h2>
          <span className="muted">{shown.length} faces{sel.size ? ` · ${sel.size} selected` : ""}</span>
          <span className="grow" />
          <input list="names" value={name} onChange={(e) => setName(e.target.value)} placeholder="Name" />
          <datalist id="names">{people.map((p) => <option key={p.id} value={p.name} />)}</datalist>
          {view.kind === "review" && <>
            <button className="ok" disabled={busy} onClick={act.accept}>Yes, the guess (Y)</button>
            <button className="warn" disabled={busy} onClick={act.reject}>No (R)</button>
            <button disabled={busy} onClick={act.moveTo}>It's this name</button>
          </>}
          {view.kind === "person" && <>
            <button className="warn" disabled={busy} onClick={act.notThem}>Not them (R)</button>
            <button disabled={busy} onClick={() => act.training(true)}>Exclude from training (T)</button>
            <button disabled={busy} onClick={() => act.training(false)}>Include</button>
            <button disabled={busy} onClick={act.moveTo} title="Moves selected faces; creates the person if the name is new">
              Move to name / new person</button>
            <button disabled={busy} onClick={act.rename}>Rename to</button>
            <button className={armDelete ? "warn primary" : "warn"} disabled={busy} onClick={act.deletePerson}>
              {armDelete ? "Confirm delete" : "Delete person"}</button>
          </>}
          {view.kind === "conflicts" && <>
            <button className="warn" disabled={busy} onClick={act.notThem}>Not them (R)</button>
            <button disabled={busy} onClick={act.moveTo}>Move to name / new person</button>
          </>}
          {view.kind === "group" && (
            <button className="primary" disabled={busy} onClick={act.nameGroup}>
              {sel.size ? "Name selected" : "Name group"}</button>)}
          {view.kind === "unsorted" && <>
            <button disabled={busy} onClick={act.moveTo}>Name selected</button>
            <button disabled={busy} onClick={act.autoFile}>Match to people</button>
          </>}
        </div>
        {note && <div className="note">{note}</div>}
        {sel.size > 0 && (
          <div className="note">
            <button className="warn" disabled={busy} onClick={act.notFace}>Not a face (F)</button>
            <span className="muted small"> for crops that aren't a face. Double-click any face to open its photo,
              where you can draw the right box.</span>
          </div>)}
        {view.kind === "conflicts" && conflicts.length > 0 &&
          <div className="note small">One person is filed on two or more faces in these photos, so at least one is wrong.
            Double-click a face to open the photo and fix it there.</div>}
        <FaceGrid faces={shown} selected={sel} setSelected={setSel} onHover={setHover} onOpen={setOpen}
          caption={view.kind === "review" ? (f) =>
            `${f.guess_person_id ? names[f.guess_person_id] : "?"} ${(f.score ?? 0).toFixed(2)}${f.second_pass ? " ·2nd" : ""}`
            : view.kind === "conflicts" ? (f) => `${f.person_id ? names[f.person_id] : ""} · ${f.filename}` : undefined} />
      </section>

      <aside className="preview">
        {hover?.thumb ? <img src={hover.thumb} alt="" /> : <div className="ph" />}
        <p className="muted small">{hover?.filename}</p>
        {hover && <button onClick={() => setOpen(hover)}>Open photo</button>}
        <div className="help">
          Click to select, Shift-click for a range, <kbd>A</kbd> all, <kbd>Esc</kbd> none.<br />
          <kbd>R</kbd> not them · <kbd>F</kbd> not a face · <kbd>T</kbd> exclude from training (filters, heavy edits) ·
          <kbd>Y</kbd> accept guess in Review.<br />
          <b>auto</b> = filed by the matcher, not by you. Removing one stops it coming back.
        </div>
      </aside>
      {open && <PhotoViewer photoId={open.photo_id} focusFaceId={open.id}
        onClose={() => { setOpen(null); refresh().then(() => loadView(view)); }} onChange={onChange} />}
    </main>
  );
}
