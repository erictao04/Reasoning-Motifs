import { NavLink } from "react-router-dom";
import type { PropsWithChildren } from "react";

export function AppShell({ children }: PropsWithChildren) {
  return (
    <div className="app-frame">
      <header className="topbar">
        <div>
          <p className="eyebrow">Reasoning Motifs</p>
          <h1>Research Explorer</h1>
        </div>
        <nav className="topnav" aria-label="Primary">
          <NavLink to="/">Story</NavLink>
          <NavLink to="/questions">Questions</NavLink>
        </nav>
      </header>
      <main>{children}</main>
    </div>
  );
}
