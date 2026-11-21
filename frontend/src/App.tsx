import { useCallback, useEffect, useState } from "react";
import { api, type Stats } from "./api";
import Library from "./pages/Library";
import People from "./pages/People";

type Tab = "library" | "people";

export default function App() {
  const [tab, setTab] = useState<Tab>(() => (location.hash.slice(1) as Tab) || "library");
  const [stats, setStats] = useState<Stats | null>(null);
  const refreshStats = useCallback(() => { api.stats().then(setStats).catch(() => {}); }, []);

  useEffect(() => { location.hash = tab; }, [tab]);
  useEffect(() => { refreshStats(); const t = setInterval(refreshStats, 5000); return () => clearInterval(t); },
    [refreshStats]);

  return (
    <div className="app">
      <header className="topbar">
        <span className="logo">Studio</span>
        <nav className="tabs">
          <button className={tab === "library" ? "on" : ""} onClick={() => setTab("library")}>
            Library {stats ? <small>{stats.photos}</small> : null}
          </button>
          <button className={tab === "people" ? "on" : ""} onClick={() => setTab("people")}>
            People {stats?.review ? <span className="pill">{stats.review}</span> : null}
          </button>
        </nav>
        <span className="muted">
          {stats ? `${stats.faces} faces · ${stats.people} people` : ""}
        </span>
      </header>
      {tab === "library" ? <Library onChange={refreshStats} /> : <People onChange={refreshStats} />}
    </div>
  );
}
