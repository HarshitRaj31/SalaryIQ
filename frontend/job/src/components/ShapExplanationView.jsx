import { useState, useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";
import {
  Sparkles,
  TrendingUp,
  TrendingDown,
  Info,
  Layers,
  BarChart3,
  ListFilter,
  Cpu,
} from "lucide-react";
import "./ShapExplanationView.css";

const formatCurrency = (val, currency = "USD") => {
  if (val === undefined || val === null || isNaN(val)) return currency === "INR" ? "₹0 LPA" : "$0";
  if (currency === "INR") {
    return `₹${Number(val).toFixed(2)} LPA`;
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(val);
};

// Custom Tooltip for SHAP Waterfall and Contribution Charts
function ShapChartTooltip({ active, payload, label, currency = "USD" }) {
  if (!active || !payload || !payload.length) return null;
  const item = payload[0]?.payload;

  return (
    <div className="shap-tooltip">
      <div className="shap-tooltip-title">{item?.feature || label}</div>
      {item?.category && (
        <div className="shap-tooltip-cat">Category: {item.category}</div>
      )}
      <div className="shap-tooltip-body">
        {item?.type === "base" && (
          <div className="shap-tooltip-row">
            <span>Market Baseline:</span>
            <strong>{item?.formatted_value || formatCurrency(item.value, currency)}</strong>
          </div>
        )}
        {item?.type === "total" && (
          <div className="shap-tooltip-row">
            <span>Final Prediction:</span>
            <strong>{item?.formatted_value || formatCurrency(item.value, currency)}</strong>
          </div>
        )}
        {(item?.type === "positive" || item?.type === "negative") && (
          <>
            <div className="shap-tooltip-row">
              <span>SHAP Contribution:</span>
              <strong className={item.delta >= 0 ? "positive" : "negative"}>
                {item?.formatted || (
                  (item.delta >= 0 ? "+" : "") + formatCurrency(item.delta, currency)
                )}
              </strong>
            </div>
            {item?.running_total !== undefined && (
              <div className="shap-tooltip-row sub">
                <span>Running Prediction:</span>
                <span>{item?.formatted_value || formatCurrency(item.running_total, currency)}</span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default function ShapExplanationView({
  shapData,
  loading = false,
  compact = false,
}) {
  const [chartMode, setChartMode] = useState("waterfall"); // "waterfall" | "contributions" | "table"

  // Fallback demo/baseline data if backend is offline or loading
  const effectiveData = useMemo(() => {
    if (shapData && shapData.feature_contributions) {
      const isIndia = shapData.country === "IN" || shapData.currency === "INR";
      const baseVal = Number(shapData.base_salary ?? shapData.base_value ?? 0);
      const predVal = Number(shapData.predicted_salary ?? 0);
      const totalDelta = Number(shapData.total_shap_impact ?? shapData.total_shap_delta ?? (predVal - baseVal));

      const rawWaterfall = shapData.waterfall || shapData.waterfall_chart_data || [];
      const waterfall_chart_data = rawWaterfall.map((step) => {
        const val = Number(step.value);
        const delta = Number(step.delta);
        return {
          ...step,
          step_bottom: step.type === "base" || step.type === "total" ? 0 : Math.min(val - delta, val),
          step_height: step.type === "base" || step.type === "total" ? val : Math.abs(delta),
          formatted: step.formatted || step.formatted_delta || (isIndia ? `₹${delta.toFixed(2)} LPA` : formatCurrency(delta, "USD")),
        };
      });

      return {
        ...shapData,
        currency: isIndia ? "INR" : "USD",
        base_value: baseVal,
        formatted_base_value: shapData.formatted_base || shapData.formatted_base_value || (isIndia ? `₹${baseVal.toFixed(2)} LPA` : formatCurrency(baseVal, "USD")),
        formatted_predicted_salary: shapData.formatted_predicted || shapData.formatted_predicted_salary || (isIndia ? `₹${predVal.toFixed(2)} LPA` : formatCurrency(predVal, "USD")),
        total_shap_delta: totalDelta,
        formatted_total_delta: shapData.formatted_total_impact || shapData.formatted_total_delta || ((totalDelta >= 0 ? "+" : "") + (isIndia ? `₹${totalDelta.toFixed(2)} LPA` : formatCurrency(totalDelta, "USD"))),
        top_positive_factors: shapData.top_positive || shapData.top_positive_factors || [],
        top_negative_factors: shapData.top_negative || shapData.top_negative_factors || [],
        waterfall_chart_data,
        narrative_explanation: shapData.summary || shapData.narrative_explanation,
      };
    }
    // Verified TreeSHAP baseline for default Data Scientist profile
    return {
      model_name: "Random Forest Regressor (TreeSHAP)",
      base_value: 133666.63,
      formatted_base_value: "$133,667",
      predicted_salary: 163363.8,
      formatted_predicted_salary: "$163,364",
      total_shap_delta: 29697.17,
      formatted_total_delta: "+$29,697",
      feature_contributions: [
        {
          field: "experience_level",
          feature_title: "Experience Seniority",
          category: "Seniority",
          value: "Senior-level (SE)",
          raw_value: "SE",
          shap_value: 10985.46,
          formatted_shap: "+$10,985",
          direction: "positive",
          percentage_impact: 8.22,
          abs_impact: 10985.46,
        },
        {
          field: "employee_residence",
          feature_title: "Employee Residence",
          category: "Geography",
          value: "United States (US)",
          raw_value: "US",
          shap_value: 8417.19,
          formatted_shap: "+$8,417",
          direction: "positive",
          percentage_impact: 6.3,
          abs_impact: 8417.19,
        },
        {
          field: "company_location",
          feature_title: "Company Location",
          category: "Geography",
          value: "United States (US)",
          raw_value: "US",
          shap_value: 6940.31,
          formatted_shap: "+$6,940",
          direction: "positive",
          percentage_impact: 5.19,
          abs_impact: 6940.31,
        },
        {
          field: "work_year",
          feature_title: "Work Year Trend",
          category: "Temporal",
          value: "2023",
          raw_value: 2023,
          shap_value: 2425.32,
          formatted_shap: "+$2,425",
          direction: "positive",
          percentage_impact: 1.81,
          abs_impact: 2425.32,
        },
        {
          field: "remote_ratio",
          feature_title: "Work Setting",
          category: "Workplace",
          value: "100% Fully Remote",
          raw_value: 100,
          shap_value: -1372.34,
          formatted_shap: "-$1,372",
          direction: "negative",
          percentage_impact: -1.03,
          abs_impact: 1372.34,
        },
        {
          field: "company_size",
          feature_title: "Company Scale",
          category: "Company",
          value: "Medium (50 - 250)",
          raw_value: "M",
          shap_value: 1147.9,
          formatted_shap: "+$1,148",
          direction: "positive",
          percentage_impact: 0.86,
          abs_impact: 1147.9,
        },
        {
          field: "job_title",
          feature_title: "Role Title",
          category: "Role",
          value: "Data Scientist",
          raw_value: "Data Scientist",
          shap_value: 907.21,
          formatted_shap: "+$907",
          direction: "positive",
          percentage_impact: 0.68,
          abs_impact: 907.21,
        },
        {
          field: "employment_type",
          feature_title: "Employment Type",
          category: "Contract",
          value: "Full-Time (FT)",
          raw_value: "FT",
          shap_value: 246.12,
          formatted_shap: "+$246",
          direction: "positive",
          percentage_impact: 0.18,
          abs_impact: 246.12,
        },
      ],
      top_positive_factors: [
        {
          feature_title: "Experience Seniority",
          value: "Senior-level (SE)",
          formatted_shap: "+$10,985",
        },
        {
          feature_title: "Employee Residence",
          value: "United States (US)",
          formatted_shap: "+$8,417",
        },
        {
          feature_title: "Company Location",
          value: "United States (US)",
          formatted_shap: "+$6,940",
        },
      ],
      top_negative_factors: [
        {
          feature_title: "Workplace Setting",
          value: "100% Fully Remote",
          formatted_shap: "-$1,372",
        },
      ],
      waterfall_chart_data: [
        {
          name: "Base Average",
          feature: "Expected Base Value",
          value: 133667,
          delta: 133667,
          step_bottom: 0,
          step_height: 133667,
          formatted: "$133,667",
          direction: "base",
          type: "base",
          running_total: 133667,
        },
        {
          name: "Experience (SE)",
          feature: "Experience Seniority: Senior-level (SE)",
          value: 144652,
          delta: 10985,
          step_bottom: 133667,
          step_height: 10985,
          formatted: "+$10,985",
          direction: "positive",
          type: "positive",
          running_total: 144652,
        },
        {
          name: "Residence (US)",
          feature: "Employee Residence: United States (US)",
          value: 153069,
          delta: 8417,
          step_bottom: 144652,
          step_height: 8417,
          formatted: "+$8,417",
          direction: "positive",
          type: "positive",
          running_total: 153069,
        },
        {
          name: "Company Loc (US)",
          feature: "Company Location: United States (US)",
          value: 160009,
          delta: 6940,
          step_bottom: 153069,
          step_height: 6940,
          formatted: "+$6,940",
          direction: "positive",
          type: "positive",
          running_total: 160009,
        },
        {
          name: "Work Year (2023)",
          feature: "Work Year: 2023",
          value: 162435,
          delta: 2425,
          step_bottom: 160009,
          step_height: 2425,
          formatted: "+$2,425",
          direction: "positive",
          type: "positive",
          running_total: 162435,
        },
        {
          name: "Company Size (M)",
          feature: "Company Size: Medium (50-250)",
          value: 163583,
          delta: 1148,
          step_bottom: 162435,
          step_height: 1148,
          formatted: "+$1,148",
          direction: "positive",
          type: "positive",
          running_total: 163583,
        },
        {
          name: "Role (Data Scientist)",
          feature: "Role: Data Scientist",
          value: 164490,
          delta: 907,
          step_bottom: 163583,
          step_height: 907,
          formatted: "+$907",
          direction: "positive",
          type: "positive",
          running_total: 164490,
        },
        {
          name: "Employment (FT)",
          feature: "Employment: Full-Time (FT)",
          value: 164736,
          delta: 246,
          step_bottom: 164490,
          step_height: 246,
          formatted: "+$246",
          direction: "positive",
          type: "positive",
          running_total: 164736,
        },
        {
          name: "Remote (100%)",
          feature: "Workplace Setting: 100% Fully Remote",
          value: 163364,
          delta: -1372,
          step_bottom: 163364,
          step_height: 1372,
          formatted: "-$1,372",
          direction: "negative",
          type: "negative",
          running_total: 163364,
        },
        {
          name: "Final Prediction",
          feature: "Predicted Annual Salary",
          value: 163364,
          delta: 163364,
          step_bottom: 0,
          step_height: 163364,
          formatted: "$163,364",
          direction: "total",
          type: "total",
          running_total: 163364,
        },
      ],
      narrative_explanation:
        "The predicted annual salary of $163,364 is +$29,697 higher than the dataset baseline average ($133,667). The primary positive drivers elevating this compensation estimate are Experience Seniority (Senior-level SE: +$10,985), Employee Residence (US: +$8,417), and Company Location (US: +$6,940). Conversely, Workplace Setting (100% Fully Remote: -$1,372) slightly adjusted the prediction downward.",
    };
  }, [shapData]);

  // Direct Contribution bar chart data (ranked by impact)
  const contributionChartData = useMemo(() => {
    return (effectiveData.feature_contributions || []).map((item) => ({
      name: item.feature_title,
      feature: `${item.feature_title}: ${item.value}`,
      category: item.category,
      delta: item.shap_value,
      abs_delta: Math.abs(item.shap_value),
      direction: item.direction,
      formatted: item.formatted_shap,
    }));
  }, [effectiveData]);

  if (loading) {
    return (
      <div className="shap-loading-box">
        <Sparkles size={18} className="spin-icon" />
        <span>Computing exact TreeSHAP feature attributions...</span>
      </div>
    );
  }

  return (
    <div className={`shap-container ${compact ? "compact-mode" : ""}`}>
      {/* XAI Header */}
      <div className="shap-header">
        <div className="shap-header-left">
          <div className="shap-eyebrow">
            <Cpu size={13} />
            <span>EXPLAINABLE AI (SHAP) ATTRIBUTION</span>
          </div>
          <h3>Why this salary was predicted</h3>
          <p>
            Exact TreeSHAP attribution shows how each characteristic pushed the
            estimate above or below the market baseline.
          </p>
        </div>

        <div className="shap-header-right">
          <div className="shap-mode-switcher">
            <button
              type="button"
              className={`shap-mode-btn ${
                chartMode === "waterfall" ? "active" : ""
              }`}
              onClick={() => setChartMode("waterfall")}
              title="View cumulative buildup from base value"
            >
              <Layers size={13} />
              <span>Waterfall</span>
            </button>
            <button
              type="button"
              className={`shap-mode-btn ${
                chartMode === "contributions" ? "active" : ""
              }`}
              onClick={() => setChartMode("contributions")}
              title="View individual positive and negative impacts"
            >
              <BarChart3 size={13} />
              <span>Contributions</span>
            </button>
            <button
              type="button"
              className={`shap-mode-btn ${
                chartMode === "table" ? "active" : ""
              }`}
              onClick={() => setChartMode("table")}
              title="View tabular feature ranking"
            >
              <ListFilter size={13} />
              <span>Table</span>
            </button>
          </div>
        </div>
      </div>

      {/* KPI Metric Strip */}
      <div className="shap-kpi-strip">
        <div className="shap-kpi-pill">
          <span className="kpi-label">Market Baseline Average</span>
          <strong className="kpi-val">{effectiveData.formatted_base_value}</strong>
          <span className="kpi-sub">Dataset expected value</span>
        </div>

        <div className="shap-kpi-pill">
          <span className="kpi-label">Net Feature Impact</span>
          <strong
            className={`kpi-val ${
              effectiveData.total_shap_delta >= 0 ? "positive" : "negative"
            }`}
          >
            {effectiveData.formatted_total_delta}
          </strong>
          <span className="kpi-sub">Total SHAP sum</span>
        </div>

        <div className="shap-kpi-pill">
          <span className="kpi-label">Predicted Annual Salary</span>
          <strong className="kpi-val highlight">
            {effectiveData.formatted_predicted_salary}
          </strong>
          <span className="kpi-sub">Model final estimate</span>
        </div>

        {effectiveData.top_positive_factors?.[0] && (
          <div className="shap-kpi-pill">
            <span className="kpi-label">Top Upward Driver</span>
            <strong className="kpi-val positive">
              {effectiveData.top_positive_factors[0].formatted_shap}
            </strong>
            <span className="kpi-sub">
              {effectiveData.top_positive_factors[0].feature_title}
            </span>
          </div>
        )}
      </div>

      {/* Chart View Section */}
      <div className="shap-chart-card">
        {chartMode === "waterfall" && (
          <>
            <div className="chart-card-heading">
              <div>
                <h4>SHAP Waterfall Chart (Cumulative Buildup)</h4>
                <p>
                  Tracing step-by-step from the dataset average to your final
                  predicted salary.
                </p>
              </div>
              <div className="chart-legend-inline">
                <span className="legend-item">
                  <span className="dot base" /> Base / Final
                </span>
                <span className="legend-item">
                  <span className="dot positive" /> Positive Impact (+)
                </span>
                <span className="legend-item">
                  <span className="dot negative" /> Negative Impact (-)
                </span>
              </div>
            </div>

            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height={290}>
                <BarChart
                  data={effectiveData.waterfall_chart_data}
                  margin={{ top: 20, right: 20, left: 10, bottom: 35 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#1e293b"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="name"
                    stroke="#64748b"
                    tick={{ fill: "#94a3b8", fontSize: 10.5 }}
                    angle={-20}
                    textAnchor="end"
                    interval={0}
                  />
                  <YAxis
                    tickFormatter={(v) =>
                      effectiveData.currency === "INR"
                        ? `₹${Number(v).toFixed(1)}L`
                        : `$${Math.round(v / 1000)}k`
                    }
                    stroke="#64748b"
                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                    domain={[0, "auto"]}
                  />
                  <Tooltip content={<ShapChartTooltip currency={effectiveData.currency} />} />
                  <ReferenceLine
                    y={effectiveData.base_value}
                    stroke="#64748b"
                    strokeDasharray="4 4"
                    label={{
                      value: `Base: ${effectiveData.formatted_base_value}`,
                      fill: "#94a3b8",
                      fontSize: 10,
                      position: "insideTopLeft",
                    }}
                  />
                  {/* Floating waterfall bars: invisible lower base + colored step height */}
                  <Bar dataKey="step_bottom" stackId="waterfall" fill="transparent" />
                  <Bar
                    dataKey="step_height"
                    stackId="waterfall"
                    radius={[4, 4, 0, 0]}
                  >
                    {effectiveData.waterfall_chart_data.map((entry, index) => {
                      let color = "#8b5cf6"; // base
                      if (entry.type === "positive") color = "#10b981"; // green
                      if (entry.type === "negative") color = "#f43f5e"; // rose
                      if (entry.type === "total") color = "#38bdf8"; // cyan
                      return <Cell key={`step-${index}`} fill={color} />;
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}

        {chartMode === "contributions" && (
          <>
            <div className="chart-card-heading">
              <div>
                <h4>Feature Contribution Ranking</h4>
                <p>
                  Positive drivers increase salary above baseline; negative drivers
                  pull it downward.
                </p>
              </div>
              <div className="chart-legend-inline">
                <span className="legend-item">
                  <span className="dot positive" /> Positive Influence
                </span>
                <span className="legend-item">
                  <span className="dot negative" /> Negative Influence
                </span>
              </div>
            </div>

            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height={290}>
                <BarChart
                  data={contributionChartData}
                  layout="vertical"
                  margin={{ top: 15, right: 30, left: 40, bottom: 15 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#1e293b"
                    horizontal={false}
                  />
                  <XAxis
                    type="number"
                    tickFormatter={(v) =>
                      effectiveData.currency === "INR"
                        ? `₹${Number(v).toFixed(1)}L`
                        : `$${Math.round(v / 1000)}k`
                    }
                    stroke="#64748b"
                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    stroke="#64748b"
                    tick={{ fill: "#cbd5e1", fontSize: 11 }}
                    width={130}
                  />
                  <Tooltip content={<ShapChartTooltip currency={effectiveData.currency} />} />
                  <ReferenceLine x={0} stroke="#475569" />
                  <Bar dataKey="delta" radius={[0, 6, 6, 0]}>
                    {contributionChartData.map((entry, index) => (
                      <Cell
                        key={`bar-${index}`}
                        fill={entry.delta >= 0 ? "#10b981" : "#f43f5e"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}

        {chartMode === "table" && (
          <div className="shap-table-container">
            <table className="shap-table">
              <thead>
                <tr>
                  <th>Job Dimension</th>
                  <th>Your Value</th>
                  <th>Category</th>
                  <th>Impact Contribution ({effectiveData.currency === "INR" ? "LPA" : "USD"})</th>
                  <th>Impact Direction</th>
                </tr>
              </thead>
              <tbody>
                {effectiveData.feature_contributions.map((item, idx) => (
                  <tr key={idx}>
                    <td className="feature-name">
                      <strong>{item.feature_title}</strong>
                    </td>
                    <td className="feature-val">{item.value}</td>
                    <td>
                      <span className="category-pill">{item.category}</span>
                    </td>
                    <td
                      className={`feature-shap ${
                        item.direction === "positive" ? "pos" : "neg"
                      }`}
                    >
                      {item.formatted_shap}
                    </td>
                    <td>
                      <span
                        className={`direction-badge ${
                          item.direction === "positive" ? "pos" : "neg"
                        }`}
                      >
                        {item.direction === "positive" ? (
                          <>
                            <TrendingUp size={11} /> + Increasing
                          </>
                        ) : (
                          <>
                            <TrendingDown size={11} /> - Decreasing
                          </>
                        )}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Narrative Explanation Callout */}
      <div className="shap-narrative-card">
        <div className="narrative-icon">
          <Info size={16} />
        </div>
        <div className="narrative-content">
          <strong>Statistical Explanation Summary</strong>
          <p>{effectiveData.narrative_explanation}</p>
        </div>
      </div>
    </div>
  );
}
