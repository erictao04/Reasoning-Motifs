type NodeTone = "input" | "process" | "output" | "check";

type PipelineNodeProps = {
  x: number;
  y: number;
  width: number;
  height: number;
  title: string;
  lines: string[];
  tone: NodeTone;
};

function PipelineNode({ x, y, width, height, title, lines, tone }: PipelineNodeProps) {
  const palette: Record<NodeTone, { fill: string; stroke: string }> = {
    input: { fill: "#f7f4ea", stroke: "#d7caa0" },
    process: { fill: "#f4f7fb", stroke: "#cfd9e6" },
    output: { fill: "#f3f8f4", stroke: "#c9dccd" },
    check: { fill: "#fbf6f2", stroke: "#decfbe" },
  };
  const colors = palette[tone];

  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={18}
        fill={colors.fill}
        stroke={colors.stroke}
        strokeWidth={1.5}
      />
      <text x={x + width / 2} y={y + 28} textAnchor="middle" className="pipeline-figure__title">
        {title}
      </text>
      {lines.map((line, index) => (
        <text
          key={`${title}-${line}`}
          x={x + width / 2}
          y={y + 52 + (index * 18)}
          textAnchor="middle"
          className="pipeline-figure__line"
        >
          {line}
        </text>
      ))}
    </g>
  );
}

type ArrowProps = {
  d: string;
};

function Arrow({ d }: ArrowProps) {
  return <path d={d} className="pipeline-figure__arrow" markerEnd="url(#pipeline-arrowhead)" />;
}

export function PipelineFigure() {
  return (
    <div className="pipeline-figure-shell">
      <svg
        viewBox="0 0 1040 520"
        role="img"
        aria-labelledby="pipeline-figure-title pipeline-figure-desc"
        className="pipeline-figure"
      >
        <title id="pipeline-figure-title">Reasoning motifs project pipeline</title>
        <desc id="pipeline-figure-desc">
          Adaptive sampling feeds a mixed-outcome trace pool. The main analysis flow then goes
          through typed tokenization, question-local sequence mining, motif cards, and a web app
          interpretability tool, with a predictive sanity check as a side branch.
        </desc>

        <defs>
          <marker
            id="pipeline-arrowhead"
            markerWidth="10"
            markerHeight="10"
            refX="8"
            refY="5"
            orient="auto"
          >
            <path d="M0,0 L10,5 L0,10 z" fill="#596273" />
          </marker>
        </defs>

        <PipelineNode
          x={48}
          y={86}
          width={220}
          height={108}
          title="Adaptive sampling"
          lines={[
            "Scout questions, then densify",
            "mixed-outcome cases",
          ]}
          tone="input"
        />
        <PipelineNode
          x={316}
          y={86}
          width={220}
          height={108}
          title="Mixed-outcome trace pool"
          lines={[
            "450 GPT-OSS traces",
            "15 questions, correct vs. incorrect",
          ]}
          tone="input"
        />
        <PipelineNode
          x={584}
          y={86}
          width={220}
          height={108}
          title="Typed tokenization"
          lines={[
            "action / strategy /",
            "milestone / noise tokens",
          ]}
          tone="process"
        />
        <PipelineNode
          x={584}
          y={248}
          width={220}
          height={116}
          title="Question-local mining"
          lines={[
            "Contiguous bigrams/trigrams",
            "within-question success/failure contrast",
          ]}
          tone="process"
        />
        <PipelineNode
          x={852}
          y={248}
          width={140}
          height={116}
          title="Motif cards"
          lines={[
            "Success motifs",
            "Failure motifs",
          ]}
          tone="output"
        />
        <PipelineNode
          x={852}
          y={52}
          width={140}
          height={116}
          title="Web app"
          lines={[
            "Browse local patterns",
            "Inspect traces",
          ]}
          tone="output"
        />
        <PipelineNode
          x={316}
          y={392}
          width={220}
          height={92}
          title="Predictive sanity check"
          lines={[
            "Motif model vs. length baseline",
          ]}
          tone="check"
        />

        <Arrow d="M268 140 H316" />
        <Arrow d="M536 140 H584" />
        <Arrow d="M694 194 V248" />
        <Arrow d="M804 306 H852" />
        <Arrow d="M922 248 V168" />
        <Arrow d="M584 430 H536" />
        <Arrow d="M694 364 V430" />

        <text x="696" y="226" textAnchor="middle" className="pipeline-figure__note">
          contrastive comparison happens within each question
        </text>
        <text x="426" y="410" textAnchor="middle" className="pipeline-figure__note">
          secondary evaluation, not the main goal
        </text>
      </svg>
    </div>
  );
}
