import { NavLink } from "react-router-dom";
import type { PropsWithChildren } from "react";

export function AppShell({ children }: PropsWithChildren) {
  return (
    <div className="app-frame">
      <header className="topbar">
        <div className="brand-block">
          <p className="eyebrow">Reasoning Motifs</p>
          <h1>Local Motif Explorer</h1>
          <p className="topbar-copy">
            Browse question-specific reasoning patterns from the completed GPT-OSS trace set.
          </p>
        </div>
        <nav className="topnav" aria-label="Primary">
          <NavLink to="/">Questions</NavLink>
        </nav>
      </header>
      <main>{children}</main>
    </div>
  );
}
