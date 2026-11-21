import { useEffect, useState } from "react";
import { api, type PhotoDetail } from "../api";

export default function PhotoViewer({ photoId, onClose }: { photoId: number; onClose: () => void }) {
  const [d, setD] = useState<PhotoDetail | null>(null);
  useEffect(() => { api.photo(photoId).then(setD); }, [photoId]);
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  return (
    <div className="modal" onClick={onClose}>
      <div className="viewer" onClick={(e) => e.stopPropagation()}>
        {d && (
          <>
            <div className="stage">
              <img src={d.preview} alt={d.filename} />
              <svg viewBox={`0 0 ${d.width} ${d.height}`} preserveAspectRatio="xMidYMid meet">
                {d.faces.map((f) => (
                  <g key={f.id}>
                    <rect x={f.box[0]} y={f.box[1]} width={f.box[2] - f.box[0]} height={f.box[3] - f.box[1]}
                      className={f.person ? "named" : f.usable ? "unknown" : "aside"} />
                    {f.person && <text x={f.box[0]} y={f.box[1] - 8}>{f.person}</text>}
                  </g>
                ))}
              </svg>
            </div>
            <div className="viewerbar">
              <strong>{d.filename}</strong>
              <span className="muted">{d.width}×{d.height} · {d.faces.length} faces · {d.path}</span>
              <button onClick={onClose}>Close</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
