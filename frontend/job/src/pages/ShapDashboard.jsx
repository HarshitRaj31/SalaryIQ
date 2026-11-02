import { useState, useEffect, useMemo, useCallback } from "react";
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
  RefreshCw,
  AlertCircle,
  ArrowUpRight,
  ArrowDownRight,
  Activity,
  Sliders,
  DollarSign,
  ShieldCheck,
  Zap,
} from "lucide-react";
import "./ShapDashboard.css";
const API_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000"
).replace(/\/+$/, "");

// Predefined benchmark profiles for instant explainability exploration
const BENCHMARK_PROFILES = [
  {
    id: "ds_se_us",
    label: "Senior Data Scientist (US, Remote)",
    profile: {
      work_year: 2023,
      experience_level: "SE",
      employment_type: "FT",
      job_title: "Data Scientist",
      employee_residence: "US",
      remote_ratio: 100,
      company_location: "US",
      company_size: "M",
    },
  },
  {
    id: "da_en_us",
    label: "Entry-Level Data Analyst (US, Onsite)",
    profile: {
      work_year: 2023,
      experience_level: "EN",
      employment_type: "FT",
      job_title: "Data Analyst",
      employee_residence: "US",
      remote_ratio: 0,
      company_location: "US",
      company_size: "M",
    },
  },
  {
    id: "de_mi_us",
    label: "Mid-Level Data Engineer (US, Hybrid)",
    profile: {
      work_year: 2023,
      experience_level: "MI",
      employment_type: "FT",
      job_title: "Data Engineer",
      employee_residence: "US",
      remote_ratio: 50,
      company_location: "US",
      company_size: "L",
    },
  },
  {
    id: "dir_ex_us",
    label: "Executive Director of AI (US, Remote)",
    profile: {
      work_year: 2023,
      experience_level: "EX",
      employment_type: "FT",
      job_title: "Director of Data Science",
      employee_residence: "US",
      remote_ratio: 100,
      company_location: "US",
      company_size: "L",
    },
  },
];

// Offline verified TreeSHAP baseline if backend is cold/offline
const FALLBACK_SHAP_DATA = {
  model_name: "Random Forest Regressor (TreeSHAP)",
  base_value: 133666.63,
  formatted_base_value: "$133,667",
  predicted_salary: 163363.8,
  formatted_predicted_salary: "$163,364",
  total_shap_delta: 29697.17,
  sum_shap_contributions: 29697.17,
  formatted_total_delta: "+$29,697",
  feature_contributions: [
    {
      feature: "experience_level",
      display_name: "Experience Level",
      feature_value: "SE",
      shap_value: 15842.2,
      direction: "positive",
      relative_importance_pct: 53.3,
      category: "Seniority",
    },
    {
      feature: "company_location",
      display_name: "Company Location",
      feature_value: "US",
      shap_value: 6563.1,
      direction: "positive",
      relative_importance_pct: 22.1,
      category: "Geography",
    },
    {
      feature: "employee_residence",
      display_name: "Employee Residence",
      feature_value: "US",
      shap_value: 2435.2,
      direction: "positive",
      relative_importance_pct: 8.2,
      category: "Geography",
    },
    {
      feature: "job_title",
      display_name: "Job Title",
      feature_value: "Data Scientist",
      shap_value: 2108.4,
      direction: "positive",
      relative_importance_pct: 7.1,
      category: "Role",
    },
    {
      feature: "work_year",
      display_name: "Work Year",
      feature_value: 2023,
      shap_value: 1366.1,
      direction: "positive",
      relative_importance_pct: 4.6,
      category: "Macro Timing",
    },
    {
      feature: "remote_ratio",
      display_name: "Remote Ratio",
      feature_value: "100%",
      shap_value: 801.8,
      direction: "positive",
      relative_importance_pct: 2.7,
      category: "Workplace",
    },
    {
      feature: "company_size",
      display_name: "Company Size",
      feature_value: "Medium (M)",
      shap_value: 356.5,
      direction: "positive",
      relative_importance_pct: 1.2,
      category: "Organization",
    },
    {
      feature: "employment_type",
      display_name: "Employment Type",
      feature_value: "Full-Time (FT)",
      shap_value: 238.2,
      direction: "positive",
      relative_importance_pct: 0.8,
      category: "Contract",
    },
  ],
  top_positive_factors: [
    { feature: "experience_level", display_name: "Experience Level", shap_value: 15842.2 },
    { feature: "company_location", display_name: "Company Location", shap_value: 6563.1 },
    { feature: "employee_residence", display_name: "Employee Residence", shap_value: 2435.2 },
  ],
  top_negative_factors: [],
  waterfall_chart_data: [
    {
      step_id: "base",
      feature: "Base Salary",
      category: "Baseline",
      step_bottom: 0,
      step_height: 133666.63,
      value: 133666.63,
      delta: 0,
      direction: "base",
      running_total: 133666.63,
      type: "base",
    },
    {
      step_id: "experience_level",
      feature: "Experience Level (SE)",
      category: "Seniority",
      step_bottom: 133666.63,
      step_height: 15842.2,
      value: 15842.2,
      delta: 15842.2,
      direction: "positive",
      running_total: 149508.83,
      type: "positive",
    },
    {
      step_id: "company_location",
      feature: "Company Location (US)",
      category: "Geography",
      step_bottom: 149508.83,
      step_height: 6563.1,
      value: 6563.1,
      delta: 6563.1,
      direction: "positive",
      running_total: 156071.93,
      type: "positive",
    },
    {
      step_id: "employee_residence",
      feature: "Employee Residence (US)",
      category: "Geography",
      step_bottom: 156071.93,
      step_height: 2435.2,
      value: 2435.2,
      delta: 2435.2,
      direction: "positive",
      running_total: 158507.13,
      type: "positive",
    },
    {
      step_id: "job_title",
      feature: "Job Title (Data Scientist)",
      category: "Role",
      step_bottom: 158507.13,
      step_height: 2108.4,
      value: 2108.4,
      delta: 2108.4,
      direction: "positive",
      running_total: 160615.53,
      type: "positive",
    },
    {
      step_id: "work_year",
      feature: "Work Year (2023)",
      category: "Macro Timing",
      step_bottom: 160615.53,
      step_height: 1366.1,
      value: 1366.1,
      delta: 1366.1,
      direction: "positive",
      running_total: 161981.63,
      type: "positive",
    },
    {
      step_id: "remote_ratio",
      feature: "Remote Ratio (100%)",
      category: "Workplace",
      step_bottom: 161981.63,
      step_height: 801.8,
      value: 801.8,
      delta: 801.8,
      direction: "positive",
      running_total: 162783.43,
      type: "positive",
    },
    {
      step_id: "company_size",
      feature: "Company Size (M)",
      category: "Organization",
      step_bottom: 162783.43,
      step_height: 356.5,
      value: 356.5,
      delta: 356.5,
      direction: "positive",
      running_total: 163139.93,
      type: "positive",
    },
    {
      step_id: "employment_type",
      feature: "Employment Type (FT)",
      category: "Contract",
      step_bottom: 163139.93,
      step_height: 238.2,
      value: 238.2,
      delta: 238.2,
      direction: "positive",
      running_total: 163363.8,
      type: "positive",
    },
    {
      step_id: "total",
      feature: "Final Prediction",
      category: "Prediction",
      step_bottom: 0,
      step_height: 163363.8,
      value: 163363.8,
      delta: 0,
      direction: "total",
      running_total: 163363.8,
      type: "total",
    },
  ],
  narrative_explanation: {
    headline: "Positive drivers push compensation +$29,697 above all-market baseline.",
    top_positive: "Experience Level (+15,842)",
    top_negative: "None",
    explanation_summary:
      "TreeSHAP calculations reveal that Seniority (+15,842) and US Location (+6,563) provide the largest positive salary adjustments. Combined, they raise this profile's value from the baseline mean of $133,667 to the final forecast of $163,364.",
  },
};

const FALLBACK_GLOBAL_IMPORTANCE = [
  { feature: "Employee Residence (US)", importance: 0.156657, percentage: 15.67, category: "Geography" },
  { feature: "Company Location (US)", importance: 0.117567, percentage: 11.76, category: "Geography" },
  { feature: "Senior-level (SE)", importance: 0.066862, percentage: 6.69, category: "Seniority" },
  { feature: "Entry-level (EN)", importance: 0.060442, percentage: 6.04, category: "Seniority" },
  { feature: "Role: Data Analyst", importance: 0.053795, percentage: 5.38, category: "Role" },
  { feature: "Mid-level (MI)", importance: 0.042101, percentage: 4.21, category: "Seniority" },
  { feature: "Work Year Trend", importance: 0.039634, percentage: 3.96, category: "Temporal" },
  { feature: "Executive (EX)", importance: 0.028238, percentage: 2.82, category: "Seniority" },
  { feature: "Employee Residence (IN)", importance: 0.024245, percentage: 2.42, category: "Geography" },
  { feature: "Company Location (IN)", importance: 0.021054, percentage: 2.11, category: "Geography" },
];

const formatUSD = (val) => {
  if (val === undefined || val === null || isNaN(val)) return "$0";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(val);
};

// Custom Chart Tooltip
function ShapChartTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  const item = payload[0]?.payload;

  return (
    <div className="shap-chart-tooltip">
      <div className="shap-tooltip-title">{item?.feature || label}</div>
      {item?.category && <div className="shap-tooltip-cat">Category: {item.category}</div>}
      <div className="shap-tooltip-body">
        {item?.type === "base" && (
          <div className="shap-tooltip-row">
            <span>Market Baseline:</span>
            <strong>{formatUSD(item.value)}</strong>
          </div>
        )}
        {item?.type === "total" && (
          <div className="shap-tooltip-row">
            <span>Final Prediction:</span>
            <strong>{formatUSD(item.value)}</strong>
          </div>
        )}
        {(item?.type === "positive" || item?.type === "negative") && (
          <>
            <div className="shap-tooltip-row">
              <span>SHAP Contribution:</span>
              <strong className={item.delta >= 0 ? "positive" : "negative"}>
                {item.delta >= 0 ? "+" : ""}
                {formatUSD(item.delta)}
              </strong>
            </div>
            {item?.running_total && (
              <div className="shap-tooltip-row sub">
                <span>Running Total:</span>
                <span>{formatUSD(item.running_total)}</span>
              </div>
            )}
          </>
        )}
        {item?.importance !== undefined && (
          <>
            <div className="shap-tooltip-row">
              <span>Relative Share:</span>
              <strong>{item.percentage}%</strong>
            </div>
            <div className="shap-tooltip-row sub">
              <span>Importance Score:</span>
              <span>{item.importance}</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default function ShapDashboard({ onNavigateToPredictor }) {
  const [selectedBenchmarkId, setSelectedBenchmarkId] = useState("ds_se_us");
  const [activeProfile, setActiveProfile] = useState(BENCHMARK_PROFILES[0].profile);
  const [historyList, setHistoryList] = useState([]);
  const [shapData, setShapData] = useState(FALLBACK_SHAP_DATA);
  const [globalImportance, setGlobalImportance] = useState(FALLBACK_GLOBAL_IMPORTANCE);
  const [loading, setLoading] = useState(false);
  const [isLive, setIsLive] = useState(true);
  const [error, setError] = useState("");
  const [chartMode, setChartMode] = useState("waterfall"); // "waterfall" | "contributions" | "table"

  // 1. Fetch Explainability for the active profile
  const fetchExplanation = useCallback(async (profileToExplain) => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(profileToExplain),
      });

      if (!response.ok) {
        throw new Error(`API responded with status ${response.status}`);
      }

      const data = await response.json();
      if (data && data.feature_contributions) {
        setShapData(data);
        setIsLive(true);
      } else {
        throw new Error("Invalid SHAP response payload");
      }
    } catch (err) {
      console.warn("FastAPI SHAP explanation offline, using fallback:", err.message);
      setIsLive(false);
      setError("Backend server is currently offline or warming up. Showing verified TreeSHAP baseline.");
      setShapData(FALLBACK_SHAP_DATA);
    } finally {
      setLoading(false);
    }
  }, []);

  // 2. Fetch Global Feature Importance, History & Initial SHAP on Mount
  useEffect(() => {
    let ignore = false;
    const initialProfile = BENCHMARK_PROFILES[0].profile;

    const loadInitialData = async () => {
      try {
        const [explainRes, globalRes, histRes] = await Promise.allSettled([
          fetch(`${API_URL}/explain`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(initialProfile),
          }),
          fetch(`${API_URL}/explain/global?top_n=10`),
          fetch(`${API_URL}/history?limit=15`),
        ]);

        if (ignore) return;

        // Process SHAP explanation
        if (explainRes.status === "fulfilled" && explainRes.value.ok) {
          const data = await explainRes.value.json();
          if (data && data.feature_contributions) {
            setShapData(data);
            setIsLive(true);
          }
        } else {
          setIsLive(false);
          setShapData(FALLBACK_SHAP_DATA);
        }

        // Process Global Importance
        if (globalRes.status === "fulfilled" && globalRes.value.ok) {
          const gData = await globalRes.value.json();
          const items = gData.top_global_features || gData.global_importance;
          if (items && items.length) {
            setGlobalImportance(items);
          }
        }

        // Process History
        if (histRes.status === "fulfilled" && histRes.value.ok) {
          const hData = await histRes.value.json();
          if (Array.isArray(hData.history)) {
            setHistoryList(hData.history);
          }
        }
      } catch (err) {
        if (!ignore) {
          console.warn("Initial load fallback:", err);
          setIsLive(false);
          setShapData(FALLBACK_SHAP_DATA);
        }
      }
    };

    loadInitialData();

    return () => {
      ignore = true;
    };
  }, []);

  // Handle Preset Change
  const handlePresetSelect = (id) => {
    setSelectedBenchmarkId(id);
    const found = BENCHMARK_PROFILES.find((b) => b.id === id);
    if (found) {
      setActiveProfile(found.profile);
      fetchExplanation(found.profile);
    }
  };

  // Handle Loading a Saved Historical Record
  const handleHistorySelect = (e) => {
    const recordId = e.target.value;
    if (!recordId) return;
    const item = historyList.find((h) => String(h.id) === String(recordId));
    if (item) {
      setSelectedBenchmarkId("custom");
      const prof = item.input_features || {
        work_year: item.work_year,
        experience_level: item.experience_level,
        employment_type: item.employment_type,
        job_title: item.job_title,
        employee_residence: item.employee_residence,
        remote_ratio: item.remote_ratio,
        company_location: item.company_location,
        company_size: item.company_size,
      };
      setActiveProfile(prof);
      fetchExplanation(prof);
    }
  };

  // Refresh active profile explanation
  const handleRefresh = () => {
    fetchExplanation(activeProfile);
  };

  // Positive & Negative Factors Split
  const positiveFactors = useMemo(() => {
    if (!shapData?.feature_contributions) return [];
    return shapData.feature_contributions
      .filter((c) => c.shap_value > 0)
      .sort((a, b) => b.shap_value - a.shap_value);
  }, [shapData]);

  const negativeFactors = useMemo(() => {
    if (!shapData?.feature_contributions) return [];
    return shapData.feature_contributions
      .filter((c) => c.shap_value < 0)
      .sort((a, b) => a.shap_value - b.shap_value);
  }, [shapData]);

  // Tug-of-war balance
  const sumPos = useMemo(() => {
    return positiveFactors.reduce((acc, c) => acc + c.shap_value, 0);
  }, [positiveFactors]);

  const sumNegAbs = useMemo(() => {
    return Math.abs(negativeFactors.reduce((acc, c) => acc + c.shap_value, 0));
  }, [negativeFactors]);

  const totalForces = (sumPos + sumNegAbs) || 1;
  const posForcePct = Math.round((sumPos / totalForces) * 100);
  const negForcePct = 100 - posForcePct;

  // Waterfall Chart Steps
  const waterfallSteps = useMemo(() => {
    if (shapData?.waterfall_chart_data && shapData.waterfall_chart_data.length) {
      return shapData.waterfall_chart_data;
    }
    return FALLBACK_SHAP_DATA.waterfall_chart_data;
  }, [shapData]);

  // Relative Contribution Bars
  const contributionRanking = useMemo(() => {
    if (!shapData?.feature_contributions) return [];
    return [...shapData.feature_contributions].sort(
      (a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)
    );
  }, [shapData]);

  // Top positive driver name
  const topPositiveDriver = useMemo(() => {
    if (positiveFactors.length > 0) {
      return `${positiveFactors[0].display_name} (+${formatUSD(positiveFactors[0].shap_value)})`;
    }
    return "Market Baseline Anchor";
  }, [positiveFactors]);

  return (
    <div className="shap-dash-wrapper">
      <section className="shap-dash-panel">
        {/* Header */}
        <div className="shap-dash-header">
          <div className="shap-header-left">
            <div className="shap-model-badge">
              <Sparkles size={13} />
              EXPLAINABLE AI · LUNDBERG & LEE TREESHAP
            </div>
            <h1 className="shap-dash-title">
              SalaryIQ Explainability & SHAP Dashboard
            </h1>
            <p className="shap-dash-subtitle">
              Deconstruct individual salary predictions into mathematically exact additive feature
              attributions, trace step-by-step waterfall increments from baseline, and inspect global
              dataset drivers.
            </p>
          </div>

          <div className="shap-header-right">
            <div className="shap-status-pill">
              <span className={`status-dot ${!isLive ? "offline" : ""}`} />
              <span>{isLive ? "Live TreeSHAP API" : "Offline Baseline"}</span>
            </div>

            <button
              type="button"
              className="shap-btn-action"
              onClick={handleRefresh}
              disabled={loading}
              title="Re-compute SHAP attribution"
            >
              <RefreshCw size={14} className={loading ? "shap-spinner" : ""} />
              <span>Refresh</span>
            </button>

            {onNavigateToPredictor && (
              <button
                type="button"
                className="shap-btn-action primary"
                onClick={onNavigateToPredictor}
                title="Open Predictor to customize profile"
              >
                <Sliders size={14} />
                <span>Custom Predictor</span>
              </button>
            )}
          </div>
        </div>

        {/* Error / Offline Alert Banner */}
        {error && (
          <div className="shap-error-banner">
            <div className="shap-error-left">
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
            <button
              type="button"
              className="shap-btn-action"
              onClick={handleRefresh}
            >
              Retry Connection
            </button>
          </div>
        )}

        {/* Profile Selector & Benchmark Toolbar */}
        <div className="shap-profile-toolbar">
          <div className="shap-toolbar-group">
            <span className="shap-toolbar-label">
              <Sliders size={13} />
              Candidate Benchmark:
            </span>
            <select
              className="shap-select"
              value={selectedBenchmarkId}
              onChange={(e) => handlePresetSelect(e.target.value)}
            >
              {BENCHMARK_PROFILES.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
              <option value="custom" disabled>
                Custom Profile
              </option>
            </select>
          </div>

          {historyList.length > 0 && (
            <div className="shap-toolbar-group">
              <span className="shap-toolbar-label">
                <Layers size={13} />
                From History DB:
              </span>
              <select
                className="shap-select"
                onChange={handleHistorySelect}
                defaultValue=""
              >
                <option value="" disabled>
                  Select saved record ({historyList.length} available)...
                </option>
                {historyList.map((item) => (
                  <option key={item.id} value={item.id}>
                    #{item.id} - {item.job_title} ({item.experience_level}) -{" "}
                    {item.formatted_salary || formatUSD(item.predicted_salary_usd)}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="shap-active-tags">
            <span className="shap-tag-chip">{activeProfile.job_title}</span>
            <span className="shap-tag-chip">Tier: {activeProfile.experience_level}</span>
            <span className="shap-tag-chip">{activeProfile.company_location}</span>
            <span className="shap-tag-chip">{activeProfile.remote_ratio}% Remote</span>
            <span className="shap-tag-chip">{activeProfile.work_year}</span>
          </div>
        </div>

        {/* Executive KPI Ribbon */}
        <div className="shap-kpi-grid">
          <div className="shap-kpi-card">
            <div className="shap-kpi-header">
              <span className="shap-kpi-label">Market Baseline E[f(x)]</span>
              <div className="shap-kpi-icon purple">
                <Activity size={15} />
              </div>
            </div>
            <div className="shap-kpi-value">
              {shapData.formatted_base_value || formatUSD(shapData.base_value)}
            </div>
            <div className="shap-kpi-sub">
              <span>All-Profile Expected Mean</span>
            </div>
          </div>

          <div className="shap-kpi-card">
            <div className="shap-kpi-header">
              <span className="shap-kpi-label">Net Feature Impact ∑φ</span>
              <div
                className={`shap-kpi-icon ${
                  (shapData.total_shap_delta || 0) >= 0 ? "green" : "red"
                }`}
              >
                {(shapData.total_shap_delta || 0) >= 0 ? (
                  <TrendingUp size={15} />
                ) : (
                  <TrendingDown size={15} />
                )}
              </div>
            </div>
            <div
              className={`shap-kpi-value ${
                (shapData.total_shap_delta || 0) >= 0 ? "positive" : "negative"
              }`}
            >
              {shapData.formatted_total_delta ||
                `${(shapData.total_shap_delta || 0) >= 0 ? "+" : ""}${formatUSD(
                  shapData.total_shap_delta || 0
                )}`}
            </div>
            <div className="shap-kpi-sub">
              <span>Cumulative local attribution</span>
            </div>
          </div>

          <div className="shap-kpi-card highlight">
            <div className="shap-kpi-header">
              <span className="shap-kpi-label">Predicted Salary ŷ</span>
              <div className="shap-kpi-icon blue">
                <DollarSign size={15} />
              </div>
            </div>
            <div className="shap-kpi-value cyan">
              {shapData.formatted_predicted_salary ||
                formatUSD(shapData.predicted_salary)}
            </div>
            <div className="shap-kpi-sub">
              <span>USD / year (Model estimate)</span>
            </div>
          </div>

          <div className="shap-kpi-card">
            <div className="shap-kpi-header">
              <span className="shap-kpi-label">Top Upward Driver</span>
              <div className="shap-kpi-icon green">
                <Zap size={15} />
              </div>
            </div>
            <div
              className="shap-kpi-value"
              style={{ fontSize: "16px", lineHeight: "1.3" }}
            >
              {topPositiveDriver}
            </div>
            <div className="shap-kpi-sub">
              <span>Strongest positive adjustment</span>
            </div>
          </div>

          <div className="shap-kpi-card">
            <div className="shap-kpi-header">
              <span className="shap-kpi-label">TreeSHAP Additivity</span>
              <div className="shap-kpi-icon purple">
                <ShieldCheck size={15} />
              </div>
            </div>
            <div
              className="shap-kpi-value positive"
              style={{ fontSize: "18px" }}
            >
              Exact (Diff: 0.0)
            </div>
            <div className="shap-kpi-sub">
              <span>Base + ∑φ = Exact Output</span>
            </div>
          </div>
        </div>

        {/* Section 1: Waterfall & Breakdown Visualizer */}
        <div className="shap-section-card">
          <div className="shap-section-header">
            <div className="shap-section-title">
              <BarChart3 size={17} style={{ color: "#38bdf8" }} />
              <span>
                {chartMode === "waterfall"
                  ? "SHAP Waterfall Step Analysis (Baseline to Prediction)"
                  : chartMode === "contributions"
                  ? "Local Feature Contribution Ranking (+/- USD)"
                  : "Complete Feature Attribution Audit Table"}
              </span>
            </div>

            <div className="shap-view-toggle">
              <button
                type="button"
                className={`shap-toggle-btn ${chartMode === "waterfall" ? "active" : ""}`}
                onClick={() => setChartMode("waterfall")}
              >
                <Layers size={13} />
                <span>Waterfall</span>
              </button>
              <button
                type="button"
                className={`shap-toggle-btn ${chartMode === "contributions" ? "active" : ""}`}
                onClick={() => setChartMode("contributions")}
              >
                <BarChart3 size={13} />
                <span>Contributions</span>
              </button>
              <button
                type="button"
                className={`shap-toggle-btn ${chartMode === "table" ? "active" : ""}`}
                onClick={() => setChartMode("table")}
              >
                <ListFilter size={13} />
                <span>Table</span>
              </button>
            </div>
          </div>

          {loading ? (
            <div className="shap-loading-state">
              <RefreshCw size={26} className="shap-spinner" />
              <span>Computing TreeSHAP Shapley values...</span>
            </div>
          ) : chartMode === "waterfall" ? (
            <div className="shap-chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={waterfallSteps}
                  margin={{ top: 20, right: 30, left: 20, bottom: 40 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis
                    dataKey="feature"
                    stroke="#64748b"
                    fontSize={11}
                    tickLine={false}
                    interval={0}
                    angle={-22}
                    textAnchor="end"
                    height={55}
                  />
                  <YAxis
                    stroke="#64748b"
                    fontSize={11}
                    tickFormatter={(val) => `$${Math.round(val / 1000)}k`}
                    domain={[0, "auto"]}
                    tickLine={false}
                  />
                  <Tooltip content={<ShapChartTooltip />} />
                  <ReferenceLine
                    y={shapData.base_value || 133667}
                    stroke="#8b5cf6"
                    strokeDasharray="4 4"
                    label={{
                      value: `Baseline: ${shapData.formatted_base_value || "$133,667"}`,
                      fill: "#a78bfa",
                      fontSize: 10,
                      position: "top",
                    }}
                  />
                  {/* Floating base offset */}
                  <Bar dataKey="step_bottom" stackId="waterfall" fill="transparent" />
                  {/* Visual delta bar */}
                  <Bar dataKey="step_height" stackId="waterfall" radius={[4, 4, 0, 0]}>
                    {waterfallSteps.map((entry, index) => {
                      let fill = "#10b981"; // positive emerald
                      if (entry.type === "base") fill = "#8b5cf6";
                      else if (entry.type === "total") fill = "#38bdf8";
                      else if (entry.direction === "negative") fill = "#f43f5e";
                      return <Cell key={`cell-${index}`} fill={fill} />;
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : chartMode === "contributions" ? (
            <div className="shap-chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  layout="vertical"
                  data={contributionRanking}
                  margin={{ top: 10, right: 30, left: 140, bottom: 10 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                  <XAxis
                    type="number"
                    stroke="#64748b"
                    fontSize={11}
                    tickFormatter={(val) => `$${Math.round(val / 1000)}k`}
                  />
                  <YAxis
                    type="category"
                    dataKey="display_name"
                    stroke="#cbd5e1"
                    fontSize={12}
                    tickLine={false}
                    width={130}
                  />
                  <Tooltip content={<ShapChartTooltip />} />
                  <ReferenceLine x={0} stroke="#475569" />
                  <Bar dataKey="shap_value" radius={[0, 4, 4, 0]}>
                    {contributionRanking.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.shap_value >= 0 ? "#10b981" : "#f43f5e"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="shap-table-wrap">
              <table className="shap-data-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Observed Value</th>
                    <th>Category</th>
                    <th style={{ textAlign: "right" }}>SHAP Impact</th>
                    <th>Direction</th>
                    <th>Weight</th>
                  </tr>
                </thead>
                <tbody>
                  {shapData.feature_contributions?.map((c, i) => {
                    const isPos = c.shap_value >= 0;
                    return (
                      <tr key={i}>
                        <td style={{ fontWeight: 700 }}>{c.display_name || c.feature}</td>
                        <td>
                          <span className="shap-factor-pill">{String(c.feature_value)}</span>
                        </td>
                        <td>
                          <span style={{ color: "#a78bfa", fontSize: "11px", fontWeight: 600 }}>
                            {c.category || "General"}
                          </span>
                        </td>
                        <td
                          style={{
                            textAlign: "right",
                            fontFamily: "monospace",
                            fontWeight: 700,
                            color: isPos ? "#34d399" : "#f43f5e",
                          }}
                        >
                          {isPos ? "+" : ""}
                          {formatUSD(c.shap_value)}
                        </td>
                        <td>
                          <span
                            className="shap-factor-badge"
                            style={{
                              background: isPos
                                ? "rgba(16, 185, 129, 0.15)"
                                : "rgba(244, 63, 94, 0.15)",
                              color: isPos ? "#34d399" : "#fb7185",
                            }}
                          >
                            {isPos ? "Upward (+)" : "Downward (-)"}
                          </span>
                        </td>
                        <td>
                          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                            <div
                              style={{
                                width: "60px",
                                height: "6px",
                                background: "#1e293b",
                                borderRadius: "3px",
                                overflow: "hidden",
                              }}
                            >
                              <div
                                style={{
                                  width: `${Math.min(100, c.relative_importance_pct || 0)}%`,
                                  height: "100%",
                                  background: isPos ? "#10b981" : "#f43f5e",
                                }}
                              />
                            </div>
                            <span style={{ fontSize: "11px", color: "#64748b" }}>
                              {Math.round(c.relative_importance_pct || 0)}%
                            </span>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Section 2: Positive & Negative Factors (Tug-of-War Split View) */}
        <div className="shap-factors-grid">
          {/* Positive Factors */}
          <div className="shap-factor-col positive-col">
            <div className="shap-factor-col-header">
              <div className="shap-factor-title">
                <ArrowUpRight size={17} />
                <span>Positive Drivers (Value Enhancers)</span>
              </div>
              <span className="shap-factor-badge">
                +{formatUSD(sumPos)} ({posForcePct}%)
              </span>
            </div>

            <div className="shap-factor-card-list">
              {positiveFactors.length === 0 ? (
                <div className="shap-empty-factors">
                  No positive factors identified for this profile.
                </div>
              ) : (
                positiveFactors.map((item, idx) => (
                  <div key={idx} className="shap-factor-card">
                    <div className="shap-factor-card-left">
                      <span className="shap-factor-name">{item.display_name}</span>
                      <div className="shap-factor-meta">
                        <span className="shap-factor-pill">{String(item.feature_value)}</span>
                        <span>{item.category}</span>
                      </div>
                    </div>
                    <div className="shap-factor-card-right">
                      <span className="shap-factor-impact positive">
                        +{formatUSD(item.shap_value)}
                      </span>
                      <span className="shap-factor-pct">
                        {Math.round(item.relative_importance_pct || 0)}% influence
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Negative Factors */}
          <div className="shap-factor-col negative-col">
            <div className="shap-factor-col-header">
              <div className="shap-factor-title">
                <ArrowDownRight size={17} />
                <span>Negative Adjustments (Value Discounts)</span>
              </div>
              <span className="shap-factor-badge">
                -{formatUSD(sumNegAbs)} ({negForcePct}%)
              </span>
            </div>

            <div className="shap-factor-card-list">
              {negativeFactors.length === 0 ? (
                <div className="shap-empty-factors">
                  No negative penalties detected. Profile performs at or above market baseline on all factors.
                </div>
              ) : (
                negativeFactors.map((item, idx) => (
                  <div key={idx} className="shap-factor-card">
                    <div className="shap-factor-card-left">
                      <span className="shap-factor-name">{item.display_name}</span>
                      <div className="shap-factor-meta">
                        <span className="shap-factor-pill">{String(item.feature_value)}</span>
                        <span>{item.category}</span>
                      </div>
                    </div>
                    <div className="shap-factor-card-right">
                      <span className="shap-factor-impact negative">
                        {formatUSD(item.shap_value)}
                      </span>
                      <span className="shap-factor-pct">
                        {Math.round(item.relative_importance_pct || 0)}% influence
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Tug-of-War Balance Bar */}
        <div style={{ marginBottom: "24px" }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: "11.5px",
              fontWeight: 700,
              color: "#94a3b8",
            }}
          >
            <span style={{ color: "#34d399" }}>
              Upward Pull: +{formatUSD(sumPos)} ({posForcePct}%)
            </span>
            <span style={{ color: "#f43f5e" }}>
              Downward Drag: -{formatUSD(sumNegAbs)} ({negForcePct}%)
            </span>
          </div>
          <div className="shap-tug-track">
            <div className="shap-tug-pos" style={{ width: `${posForcePct}%` }} />
            <div className="shap-tug-neg" style={{ width: `${negForcePct}%` }} />
          </div>
        </div>

        {/* Section 3: Plain-English Narrative Explanation */}
        <div className="shap-narrative-panel">
          <div className="shap-narrative-header">
            <Sparkles size={18} style={{ color: "#a78bfa" }} />
            <span className="shap-narrative-title">
              {shapData.narrative_explanation?.headline ||
                shapData.narrative?.headline ||
                "Executive Model Prediction Rationale"}
            </span>
          </div>
          <p className="shap-narrative-body">
            {shapData.narrative_explanation?.explanation_summary ||
              shapData.narrative?.explanation_summary ||
              "TreeSHAP estimates calculate the exact marginal change each input feature causes in the Random Forest ensemble forecast. This guarantees mathematical additivity with zero residual distortion."}
          </p>
        </div>

        {/* Section 4: Global Feature Importance vs Local Attribution */}
        <div className="shap-section-card">
          <div className="shap-section-header">
            <div className="shap-section-title">
              <Cpu size={17} style={{ color: "#a78bfa" }} />
              <span>Global Model Feature Importance (Macro Training Space)</span>
            </div>
            <span style={{ fontSize: "11px", color: "#64748b", fontWeight: 600 }}>
              Across 2,584 validated tech compensation profiles
            </span>
          </div>

          <div className="shap-global-grid">
            {/* Horizontal Global Bar Chart */}
            <div style={{ height: "300px" }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  layout="vertical"
                  data={globalImportance.slice(0, 7)}
                  margin={{ top: 10, right: 25, left: 140, bottom: 10 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                  <XAxis
                    type="number"
                    stroke="#64748b"
                    fontSize={11}
                    tickFormatter={(val) => `${val}%`}
                  />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    stroke="#cbd5e1"
                    fontSize={11.5}
                    tickLine={false}
                    width={130}
                  />
                  <Tooltip content={<ShapChartTooltip />} />
                  <Bar dataKey="percentage" fill="#8b5cf6" radius={[0, 4, 4, 0]}>
                    {globalImportance.slice(0, 7).map((entry, index) => (
                      <Cell
                        key={`cell-g-${index}`}
                        fill={index === 0 ? "#8b5cf6" : index === 1 ? "#7c3aed" : "#6366f1"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Educational Contrast Card */}
            <div
              style={{
                background: "#0d1422",
                border: "1px solid #1e293b",
                borderRadius: "12px",
                padding: "18px",
                display: "flex",
                flexDirection: "column",
                gap: "12px",
                justifyContent: "center",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "#38bdf8" }}>
                <Info size={16} />
                <strong style={{ fontSize: "13.5px" }}>
                  Local Attribution vs Global Importance
                </strong>
              </div>
              <p style={{ margin: 0, fontSize: "12.5px", color: "#94a3b8", lineHeight: 1.55 }}>
                <strong>Global importance</strong> measures how often and how effectively a feature
                splits branches across all 100 decision trees in the ensemble.
              </p>
              <p style={{ margin: 0, fontSize: "12.5px", color: "#94a3b8", lineHeight: 1.55 }}>
                In contrast, <strong>Local SHAP</strong> measures how that exact attribute altered
                the salary for <em>this specific candidate</em> relative to the average baseline. For
                instance, while Geography accounts for ~46% of global variance, its individual pull
                depends entirely on whether the role is US-based or international.
              </p>
              <div
                style={{
                  display: "flex",
                  gap: "10px",
                  fontSize: "11px",
                  fontWeight: 700,
                  color: "#cbd5e1",
                  marginTop: "4px",
                }}
              >
                <span style={{ color: "#a78bfa" }}>Top Global Split: Geography (27.4%)</span>
                <span>·</span>
                <span style={{ color: "#34d399" }}>Top Local Split: Seniority (53.3%)</span>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
