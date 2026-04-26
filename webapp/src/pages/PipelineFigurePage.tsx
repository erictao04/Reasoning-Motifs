import { PipelineFigure } from "../components/PipelineFigure";

export function PipelineFigurePage() {
  return (
    <section className="stack-xl">
      <section className="hero">
        <p className="eyebrow">Pipeline Figure</p>
        <h2>A frontend-rendered view of the project pipeline.</h2>
        <p className="lede">
          This diagram is drawn with React and SVG so we can iterate on the paper figure inside the
          app, keep it vector-clean, and reuse the same structure in demos or exports later.
        </p>
      </section>

      <section className="panel">
        <PipelineFigure />
      </section>

      <section className="panel">
        <div className="use-grid">
          <article className="use-card">
            <h4>Main story</h4>
            <p>
              The main line runs from mixed-outcome traces to tokenization, local motif mining,
              motif cards, and the web app.
            </p>
          </article>
          <article className="use-card">
            <h4>Interpretability first</h4>
            <p>
              The diagram emphasizes that the app and motif cards are the primary output. Prediction
              is shown as a side branch, not the central contribution.
            </p>
          </article>
          <article className="use-card">
            <h4>Paper-ready asset</h4>
            <p>
              Because the figure is SVG-based, we can later export it as a crisp vector asset for
              the report instead of maintaining a separate hand-drawn diagram.
            </p>
          </article>
        </div>
      </section>
    </section>
  );
}
