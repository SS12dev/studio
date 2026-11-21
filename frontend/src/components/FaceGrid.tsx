import { useRef } from "react";
import type { FaceItem } from "../api";

type Props = {
  faces: FaceItem[];
  selected: Set<number>;
  setSelected: (s: Set<number>) => void;
  onHover: (f: FaceItem) => void;
  caption?: (f: FaceItem) => string;
};

export default function FaceGrid({ faces, selected, setSelected, onHover, caption }: Props) {
  const last = useRef<number | null>(null);
  const click = (i: number, e: React.MouseEvent) => {
    const next = new Set(selected);
    if (e.shiftKey && last.current !== null) {
      const [a, b] = [Math.min(last.current, i), Math.max(last.current, i)];
      for (let k = a; k <= b; k++) next.add(faces[k].id);
    } else if (next.has(faces[i].id)) next.delete(faces[i].id);
    else next.add(faces[i].id);
    last.current = i;
    setSelected(next);
  };
  if (!faces.length) return <div className="empty">Nothing here.</div>;
  return (
    <div className="faces">
      {faces.map((f, i) => (
        <figure key={f.id} className={`${selected.has(f.id) ? "sel" : ""} ${f.source === "auto" ? "auto" : ""}`}
          onClick={(e) => click(i, e)} onMouseEnter={() => onHover(f)}>
          <img src={f.crop} loading="lazy" alt="" />
          {f.no_training && <span className="tag red">no train</span>}
          {f.source === "auto" && <span className="tag right">auto</span>}
          {!f.usable && <span className="tag low">{f.reject_reason?.replace("_", " ")}</span>}
          <figcaption>{caption ? caption(f) : f.filename}</figcaption>
        </figure>
      ))}
    </div>
  );
}
