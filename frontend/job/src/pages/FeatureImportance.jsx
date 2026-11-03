import { useState, useEffect, useMemo, useCallback } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  CartesianGrid,
} from "recharts";
import {
  Cpu,
  RefreshCw,
  Sparkles,
  Info,
  AlertCircle,
  TrendingUp,
  Layers,
  Filter,
} from "lucide-react";
import "./FeatureImportance.css";

const API_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000"
).replace(/\/+$/, "");

// Pre-computed verified Random Forest feature importance baseline from trained model
const FALLBACK_RF_FEATURES = [
  {
    feature: "Employee Residence: United States (US)",
    raw_feature: "cat__employee_residence_US",
    importance: 0.156657,
    percentage: 15.67,
    category: "Geography",
  },
  {
    feature: "Company Location: United States (US)",
    raw_feature: "cat__company_location_US",
    importance: 0.117567,
    percentage: 11.76,
    category: "Geography",
  },
  {
    feature: "Experience: Senior-level (SE)",
    raw_feature: "cat__experience_level_SE",
    importance: 0.066862,
    percentage: 6.69,
    category: "Experience",
  },
  {
    feature: "Experience: Entry-level (EN)",
    raw_feature: "cat__experience_level_EN",
    importance: 0.060442,
    percentage: 6.04,
    category: "Experience",
  },
  {
    feature: "Role: Data Analyst",
    raw_feature: "cat__job_title_Data Analyst",
    importance: 0.053795,
    percentage: 5.38,
    category: "Role",
  },
  {
    feature: "Experience: Mid-level (MI)",
    raw_feature: "cat__experience_level_MI",
    importance: 0.042101,
    percentage: 4.21,
    category: "Experience",
  },
  {
    feature: "Work Year (Historical Trend)",
    raw_feature: "num__work_year",
    importance: 0.039634,
    percentage: 3.96,
    category: "Temporal",
  },
  {
    feature: "Experience: Executive (EX)",
    raw_feature: "cat__experience_level_EX",
    importance: 0.028238,
    percentage: 2.82,
    category: "Experience",
  },
  {
    feature: "Employee Residence: India (IN)",
    raw_feature: "cat__employee_residence_IN",
    importance: 0.024245,
    percentage: 2.42,
    category: "Geography",
  },
  {
    feature: "Company Location: India (IN)",
    raw_feature: "cat__company_location_IN",
    importance: 0.021054,
    percentage: 2.11,
    category: "Geography",
  },
];

const DEFAULT_EXPLANATION =
  "According to the trained Random Forest regressor, Employee Residence: United States (US) is the single most decisive factor, accounting for 15.67% of the model's split decisions, closely followed by Company Location: United States (US) (11.76%). Combined, geographic location and seniority tier (Senior SE & Entry EN) drive over 40% of tech compensation variance, demonstrating that where an employer is based and career level outweigh specific domain titles.";

const FALLBACK_INDIA_FEATURES = [
  {
    feature: "Experience in Years (experience_avg_yrs)",
    raw_feature: "num__experience_avg_yrs",
    importance: 0.6687,
    percentage: 66.87,
    category: "Experience",
  },
  {
    feature: "Workplace Setting (remote_ratio)",
    raw_feature: "num__remote_ratio",
    importance: 0.0876,
    percentage: 8.76,
    category: "Workplace",
  },
  {
    feature: "Skill Repertoire Count",
    raw_feature: "num__skills_count",
    importance: 0.0524,
    percentage: 5.24,
    category: "Experience",
  },
  {
    feature: "Employer Glassdoor Rating",
    raw_feature: "num__company_rating",
    importance: 0.0514,
    percentage: 5.14,
    category: "Company",
  },
  {
    feature: "Hiring Year Trend",
    raw_feature: "num__work_year",
    importance: 0.0445,
    percentage: 4.45,
    category: "Temporal",
  },
  {
    feature: "Job Role Specialization",
    raw_feature: "cat__role_standard",
    importance: 0.0381,
    percentage: 3.81,
    category: "Role",
  },
  {
    feature: "Metro Tech Hub (primary_city_clean)",
    raw_feature: "cat__primary_city_clean",
    importance: 0.0245,
    percentage: 2.45,
    category: "Geography",
  },
  {
    feature: "Seniority Tier (experience_level)",
    raw_feature: "cat__experience_level",
    importance: 0.0182,
    percentage: 1.82,
    category: "Experience",
  },
  {
    feature: "Company Size Scale",
    raw_feature: "cat__company_size",
    importance: 0.0098,
    percentage: 0.98,
    category: "Company",
  },
  {
    feature: "Contract Employment Type",
    raw_feature: "cat__employment_type",
    importance: 0.0048,
    percentage: 0.48,
    category: "Role",
  },
];

const DEFAULT_INDIA_EXPLANATION =
  "In the India tech compensation market, Years of Experience (experience_avg_yrs) is overwhelmingly the most decisive factor, driving ~66.87% of the Gradient Boosting model's variance. Workplace setting (remote_ratio at 8.76%) and quantified skill depth (skills_count at 5.24%) form the secondary tier, while primary tech hubs (Bengaluru, Hyderabad, Pune) command premium local multipliers.";

const getCategoryColor = (category) => {
  switch (category) {
    case "Geography":
      return "#3b82f6"; // Blue
    case "Experience":
      return "#8b5cf6"; // Purple
    case "Role":
      return "#10b981"; // Emerald
    case "Temporal":
      return "#f59e0b"; // Amber
    case "Company":
      return "#ec4899"; // Pink
    case "Workplace":
      return "#06b6d4"; // Cyan
    default:
      return "#6366f1"; // Indigo
  }
};

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const catColor = getCategoryColor(data.category);

    return (
      <div className="fi-custom-tooltip">
        <div className="fi-tooltip-header">{data.feature}</div>
        <div className="fi-tooltip-row">
          <span className="fi-tooltip-label">Category:</span>
          <span className="fi-tooltip-val" style={{ color: catColor }}>
            {data.category}
          </span>
        </div>
        <div className="fi-tooltip-row">
          <span className="fi-tooltip-label">Predictive Impact:</span>
          <span className="fi-tooltip-val" style={{ color: "#38bdf8" }}>
            {data.percentage}%
          </span>
        </div>
        <div className="fi-tooltip-row">
          <span className="fi-tooltip-label">Importance Score:</span>
          <span className="fi-tooltip-val">
            {typeof data.importance === "number"
              ? data.importance.toFixed(4)
              : (data.percentage / 100).toFixed(4)}
          </span>
        </div>
        {data.raw_feature && (
          <div className="fi-tooltip-row" style={{ marginTop: 6 }}>
            <span className="fi-tooltip-label">Pipeline Feature:</span>
            <code style={{ fontSize: 10.5, color: "#94a3b8" }}>
              {data.raw_feature}
            </code>
          </div>
        )}
      </div>
    );
  }
  return null;
};

export default function FeatureImportance({ standalone = false }) {
  const [selectedCountry, setSelectedCountry] = useState("US"); // "US" | "IN"
  const [features, setFeatures] = useState(FALLBACK_RF_FEATURES);
  const [modelName, setModelName] = useState("Random Forest Regressor");
  const [explanation, setExplanation] = useState(DEFAULT_EXPLANATION);
  const [totalEvaluated, setTotalEvaluated] = useState(242);
  const [isLive, setIsLive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errorNotice, setErrorNotice] = useState(null);

  // User interactive state
  const [topCount, setTopCount] = useState(10);
  const [categoryFilter, setCategoryFilter] = useState("ALL");
  const [refreshCounter, setRefreshCounter] = useState(0);

  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();

    const loadData = async () => {
      try {
        let response = await fetch(
          `${API_URL}/feature-importance?country=${selectedCountry}&top_n=${topCount}`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          response = await fetch(
            `${API_URL}/api/v1/prediction/feature-importance?country=${selectedCountry}&top_n=${topCount}`,
            { signal: controller.signal }
          );
        }

        if (!response.ok) {
          throw new Error(`Server returned HTTP ${response.status}`);
        }

        const data = await response.json();

        if (isMounted) {
          if (data && data.top_features && data.top_features.length > 0) {
            setFeatures(data.top_features);
            if (data.model_name) setModelName(data.model_name);
            if (data.summary_explanation) setExplanation(data.summary_explanation);
            if (data.total_features_evaluated)
              setTotalEvaluated(data.total_features_evaluated);
            setIsLive(true);
            setErrorNotice(null);
          } else {
            throw new Error("Invalid payload format received from backend");
          }
        }
      } catch (err) {
        if (!isMounted || err.name === "AbortError") return;
        console.warn("FastAPI live feature-importance unreachable, using verified model data:", err);
        setIsLive(false);
        setErrorNotice(
          "Backend offline or unreachable. Displaying verified model pipeline snapshot."
        );
        if (selectedCountry === "IN") {
          setFeatures(FALLBACK_INDIA_FEATURES.slice(0, topCount));
          setModelName("Gradient Boosting Regressor (India Model)");
          setExplanation(DEFAULT_INDIA_EXPLANATION);
          setTotalEvaluated(10);
        } else {
          setFeatures(FALLBACK_RF_FEATURES.slice(0, topCount));
          setModelName("Random Forest Regressor (US Model)");
          setExplanation(DEFAULT_EXPLANATION);
          setTotalEvaluated(242);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    loadData();

    return () => {
      isMounted = false;
      controller.abort();
    };
  }, [selectedCountry, topCount, refreshCounter]);

  const handleRefresh = useCallback(() => {
    setLoading(true);
    setRefreshCounter((prev) => prev + 1);
  }, []);

  const handleTopCountChange = useCallback((count) => {
    setLoading(true);
    setTopCount(count);
  }, []);

  // Unique categories list
  const availableCategories = useMemo(() => {
    const cats = new Set(features.map((f) => f.category));
    return ["ALL", ...Array.from(cats)];
  }, [features]);

  // Filtered features
  const filteredFeatures = useMemo(() => {
    if (categoryFilter === "ALL") return features;
    return features.filter((f) => f.category === categoryFilter);
  }, [features, categoryFilter]);

  // Horizontal bar chart data formatted for Recharts (reversed so #1 is on top)
  const chartData = useMemo(() => {
    return filteredFeatures
      .map((item) => ({
        ...item,
        displayName:
          item.feature.length > 25
            ? item.feature.slice(0, 23) + "…"
            : item.feature,
      }))
      .slice()
      .reverse();
  }, [filteredFeatures]);

  const maxPercentage = useMemo(() => {
    if (!features.length) return 18;
    const maxVal = Math.max(...features.map((f) => f.percentage));
    return Math.ceil(maxVal + 2);
  }, [features]);

  return (
    <div className={`feature-importance-wrapper ${standalone ? "standalone-page" : ""}`}>
      <section className="feature-importance-panel">
        {/* Header */}
        <div className="feature-panel-header">
          <div className="header-left-col">
            <div className="model-badge">
              <Cpu size={13} />
              <span>{modelName}</span>
            </div>
            <h2 className="feature-panel-title">
              Model Feature Importance (Top {topCount} Attributes · {selectedCountry === "IN" ? "India 🇮🇳" : "United States 🇺🇸"})
            </h2>
            <p className="feature-panel-subtitle">
              Mathematical feature contribution derived directly from our trained{" "}
              {selectedCountry === "IN" ? "India Gradient Boosting" : "United States Random Forest"}{" "}
              regressor pipeline — revealing which job attributes exert the strongest
              influence over {selectedCountry === "IN" ? "INR (LPA)" : "USD/year"} salary predictions.
            </p>
          </div>

          <div className="header-actions-col">
            <div className="model-live-chip">
              <span className={`live-dot ${isLive ? "active" : "offline"}`} />
              <span>
                {isLive ? "Live from FastAPI Backend" : "Trained Pipeline Snapshot"}
              </span>
            </div>

            <button
              className="refresh-button"
              onClick={handleRefresh}
              disabled={loading}
              title="Refresh feature weights from backend"
            >
              <RefreshCw size={13} className={loading ? "spin-icon" : ""} />
              <span>{loading ? "Fetching..." : "Refresh"}</span>
            </button>
          </div>
        </div>

        {/* Country Selector Toggle */}
        <div className="country-selector-wrap" style={{ marginBottom: 16 }}>
          <div className="country-selector-tabs">
            <button
              type="button"
              className={`country-toggle-btn ${selectedCountry === "US" ? "active" : ""}`}
              onClick={() => {
                setLoading(true);
                setSelectedCountry("US");
              }}
            >
              <span className="country-flag-icon">🇺🇸</span>
              <span>United States (Random Forest · USD)</span>
            </button>
            <button
              type="button"
              className={`country-toggle-btn ${selectedCountry === "IN" ? "active" : ""}`}
              onClick={() => {
                setLoading(true);
                setSelectedCountry("IN");
              }}
            >
              <span className="country-flag-icon">🇮🇳</span>
              <span>India (Gradient Boosting · INR LPA)</span>
            </button>
          </div>
        </div>

        {/* Controls Bar */}
        <div className="feature-controls-bar">
          <div className="control-group">
            <span className="control-label">Limit:</span>
            <div className="segmented-toggle">
              <button
                className={`toggle-btn ${topCount === 5 ? "active" : ""}`}
                onClick={() => handleTopCountChange(5)}
              >
                Top 5
              </button>
              <button
                className={`toggle-btn ${topCount === 10 ? "active" : ""}`}
                onClick={() => handleTopCountChange(10)}
              >
                Top 10
              </button>
              <button
                className={`toggle-btn ${topCount === 15 ? "active" : ""}`}
                onClick={() => handleTopCountChange(15)}
              >
                Top 15
              </button>
            </div>
          </div>

          <div className="control-group">
            <span className="control-label">
              <Filter size={12} style={{ marginRight: 4, verticalAlign: "middle" }} />
              Category:
            </span>
            <div className="filter-pills">
              {availableCategories.map((cat) => (
                <button
                  key={cat}
                  className={`pill-filter ${categoryFilter === cat ? "active" : ""}`}
                  onClick={() => setCategoryFilter(cat)}
                >
                  {cat}
                </button>
              ))}
            </div>
          </div>
        </div>

        {errorNotice && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "10px 14px",
              background: "rgba(245, 158, 11, 0.1)",
              border: "1px solid rgba(245, 158, 11, 0.3)",
              borderRadius: 10,
              fontSize: 12.5,
              color: "#fbbf24",
              marginBottom: 20,
            }}
          >
            <AlertCircle size={15} />
            <span>{errorNotice}</span>
          </div>
        )}

        {/* 2-Column Grid: Horizontal Chart + Ranked Breakdown List */}
        <div className="feature-importance-grid">
          {/* Left Column: Horizontal Bar Chart */}
          <div className="feature-chart-subpanel">
            <div className="subpanel-title">
              <span>Horizontal Importance Chart (% Predictive Weight)</span>
              <span className="subpanel-badge">
                <TrendingUp size={12} style={{ marginRight: 4, verticalAlign: "middle" }} />
                Gini Impurity Split %
              </span>
            </div>

            <div className="chart-container" style={{ height: Math.max(380, filteredFeatures.length * 36) }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData}
                  layout="vertical"
                  margin={{ top: 10, right: 30, left: 180, bottom: 10 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#1e293b"
                    horizontal={false}
                  />
                  <XAxis
                    type="number"
                    stroke="#94a3b8"
                    tick={{ fill: "#94a3b8", fontSize: 11.5 }}
                    unit="%"
                    domain={[0, maxPercentage]}
                  />
                  <YAxis
                    type="category"
                    dataKey="displayName"
                    stroke="#94a3b8"
                    tick={{ fill: "#cbd5e1", fontSize: 11.5 }}
                    width={175}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar
                    dataKey="percentage"
                    name="Predictive Impact"
                    radius={[0, 6, 6, 0]}
                    barSize={18}
                  >
                    {chartData.map((entry, index) => (
                      <Cell
                        key={`fi-cell-${index}`}
                        fill={getCategoryColor(entry.category)}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Right Column: Ranked Top Features List */}
          <div className="feature-list-subpanel">
            <div className="subpanel-title">
              <span>Top Influential Attributes Breakdown</span>
              <span className="subpanel-badge">
                {filteredFeatures.length} of {features.length} features
              </span>
            </div>

            <div className="feature-importance-scroll-list">
              {filteredFeatures.map((item, idx) => {
                const catColor = getCategoryColor(item.category);
                return (
                  <div key={item.raw_feature || idx} className="feature-rank-row">
                    <div className="rank-number-pill">#{idx + 1}</div>
                    <div className="feature-main-detail">
                      <div className="feature-row-top">
                        <span className="feature-title-text">{item.feature}</span>
                        <span
                          className="feature-pct-tag"
                          style={{
                            color: catColor,
                            borderColor: `${catColor}55`,
                          }}
                        >
                          {item.percentage}%
                        </span>
                      </div>
                      <div className="feature-progress-track">
                        <div
                          className="feature-progress-fill"
                          style={{
                            width: `${Math.min(100, (item.percentage / 16) * 100)}%`,
                            backgroundColor: catColor,
                          }}
                        />
                      </div>
                      <div className="feature-category-sub">
                        <span>
                          Category: <strong>{item.category}</strong>
                        </span>
                        <span>
                          Weight:{" "}
                          {typeof item.importance === "number"
                            ? item.importance.toFixed(4)
                            : (item.percentage / 100).toFixed(4)}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Short Explanation of Results Box */}
        <div className="feature-explanation-box">
          <div className="explanation-header">
            <Info size={18} className="explanation-icon" />
            <strong>What do these results mean?</strong>
          </div>
          <p className="explanation-body">{explanation}</p>
          {selectedCountry === "IN" ? (
            <div className="explanation-highlights-grid">
              <div className="highlight-item">
                <span className="highlight-pill purple">1. Experience Dominance (~66.9%)</span>
                <p>
                  Years of experience is the single overwhelming driver of tech salaries in India,
                  dictating career progression and baseline LPA brackets.
                </p>
              </div>
              <div className="highlight-item">
                <span className="highlight-pill blue">2. Remote & Skill Premium</span>
                <p>
                  Workplace mode (8.8%) and quantified skill depth (5.2%) provide the strongest
                  leverage for upward compensation jumps.
                </p>
              </div>
              <div className="highlight-item">
                <span className="highlight-pill emerald">3. Metro Hub Multipliers</span>
                <p>
                  Tier-1 tech hubs (Bengaluru, Hyderabad, Gurgaon, Pune) command premium local
                  multipliers over non-metro locations.
                </p>
              </div>
            </div>
          ) : (
            <div className="explanation-highlights-grid">
              <div className="highlight-item">
                <span className="highlight-pill blue">1. Geographic Boundary Dominance</span>
                <p>
                  US employee residency and company location account for{" "}
                  <strong>~27.4%</strong> combined weight. Cross-border wage differentials
                  remain the primary driver of baseline compensation.
                </p>
              </div>
              <div className="highlight-item">
                <span className="highlight-pill purple">2. Seniority Supersedes Title</span>
                <p>
                  Moving from Entry (6.0%) to Senior (6.7%) triggers an inflection point in tree
                  splitting, proving career level matters more than subtle variations in domain
                  titles.
                </p>
              </div>
              <div className="highlight-item">
                <span className="highlight-pill amber">3. Role & Historical Trend</span>
                <p>
                  The <em>Data Analyst</em> role (5.4%) acts as an anchor for baseline analytics
                  pay, while <em>Work Year</em> (4.0%) captures macro compensation inflation from
                  2020 through 2023.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer Metadata Strip */}
        <div className="fi-footer-meta">
          <div className="meta-badges">
            <span className="meta-chip">
              <Layers size={11} style={{ marginRight: 4, verticalAlign: "middle" }} />
              Total Evaluated Features: {totalEvaluated}
            </span>
            <span className="meta-chip">Criterion: MDI / Gini Reduction</span>
            <span className="meta-chip">Trained Estimators: 100 Trees</span>
          </div>
          <div>
            <Sparkles size={11} style={{ marginRight: 4, verticalAlign: "middle" }} />
            Calculated directly from scikit-learn pipeline
          </div>
        </div>
      </section>
    </div>
  );
}
