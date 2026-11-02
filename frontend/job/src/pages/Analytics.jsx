import { useState, useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  Legend,
} from "recharts";
import {
  TrendingUp,
  DollarSign,
  Globe,
  Award,
  Sparkles,
  Filter,
  ArrowUpRight,
  Laptop,
  CheckCircle2,
  Sliders,
  Building2,
} from "lucide-react";
import FeatureImportance from "./FeatureImportance";
import "./Analytics.css";

// --- Benchmark Datasets & Metrics derived from 2,584 verified ML training records ---
const EXPERIENCE_DATA = [
  { tier: "Entry (EN)", min: 15000, median: 61900, mean: 72649, max: 130000, count: 270, growth: "Baseline" },
  { tier: "Mid-level (MI)", min: 45000, median: 95000, mean: 101829, max: 175000, count: 664, growth: "+53.5%" },
  { tier: "Senior (SE)", min: 85000, median: 148800, mean: 153897, max: 245000, count: 1554, growth: "+56.6%" },
  { tier: "Executive (EX)", min: 110000, median: 189618, mean: 191078, max: 320000, count: 96, growth: "+27.4%" },
];

const TOP_ROLES_DATA = [
  { role: "Principal Data Scientist", median: 215000, mean: 228000, count: 48, category: "Leadership" },
  { role: "Data Architect", median: 165500, mean: 168200, count: 101, category: "Engineering" },
  { role: "Applied Scientist", median: 160000, mean: 165400, count: 58, category: "Science" },
  { role: "ML Engineer", median: 150000, mean: 154800, count: 289, category: "Engineering" },
  { role: "Data Science Manager", median: 152000, mean: 156200, count: 58, category: "Leadership" },
  { role: "Data Scientist", median: 140000, mean: 144500, count: 840, category: "Science" },
  { role: "Data Engineer", median: 135000, mean: 138900, count: 1040, category: "Engineering" },
  { role: "Analytics Engineer", median: 138000, mean: 142100, count: 103, category: "Engineering" },
  { role: "Data Analyst", median: 105000, mean: 108400, count: 612, category: "Analytics" },
  { role: "BI Developer", median: 98000, mean: 102500, count: 75, category: "Analytics" },
];

const HISTORICAL_TREND_DATA = [
  { year: "2020", avgSalary: 92302, medianSalary: 85000, seniorSalary: 110000 },
  { year: "2021", avgSalary: 94085, medianSalary: 88500, seniorSalary: 118000 },
  { year: "2022", avgSalary: 124522, medianSalary: 120000, seniorSalary: 145000 },
  { year: "2023", avgSalary: 137570, medianSalary: 135000, seniorSalary: 156000 },
];

const REMOTE_DISTRIBUTION = [
  { name: "Fully Remote (100%)", value: 1643, percentage: 63.6, median: 135000, color: "#8b5cf6" },
  { name: "On-site (0%)", value: 752, percentage: 29.1, median: 130000, color: "#3b82f6" },
  { name: "Hybrid (50%)", value: 189, percentage: 7.3, median: 68000, color: "#10b981" },
];

const COMPANY_SIZE_DATA = [
  { size: "Small (< 50)", label: "Startups", median: 75000, mean: 84500, share: "5.7%", color: "#f59e0b" },
  { size: "Medium (50-250)", label: "Mid-Scale", median: 135000, mean: 141800, share: "83.1%", color: "#8b5cf6" },
  { size: "Large (> 250)", label: "Enterprise", median: 120000, mean: 128400, share: "11.2%", color: "#3b82f6" },
];

const PERCENTILE_MARKS = [
  { percentile: "10th", value: 50602, label: "Entry Offshore / Junior" },
  { percentile: "25th (Q1)", value: 84975, label: "Lower Quartile" },
  { percentile: "50th (Median)", value: 130000, label: "Market Midpoint" },
  { percentile: "75th (Q3)", value: 175000, label: "Senior Target" },
  { percentile: "90th", value: 220000, label: "Top 10% Industry Earning" },
  { percentile: "95th", value: 249470, label: "High Performance Elite" },
];

// Helper to format currency
const formatUSD = (val) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(val);

// Custom Recharts Glassmorphism Tooltip
function CustomTooltip({ active, payload, label }) {
  if (active && payload && payload.length) {
    return (
      <div className="analytics-tooltip">
        <div className="tooltip-header">{label}</div>
        {payload.map((item, idx) => (
          <div key={idx} className="tooltip-row">
            <span className="tooltip-indicator" style={{ background: item.color || item.fill }} />
            <span className="tooltip-name">{item.name}:</span>
            <strong className="tooltip-value">
              {typeof item.value === "number" && item.value > 1000
                ? formatUSD(item.value)
                : typeof item.value === "number"
                ? `${item.value.toFixed(2)}%`
                : item.value}
            </strong>
          </div>
        ))}
      </div>
    );
  }
  return null;
}

export default function Analytics({ onNavigateToPredictor }) {
  const [selectedRoleFilter, setSelectedRoleFilter] = useState("All");
  const [userBenchmarkSalary, setUserBenchmarkSalary] = useState(130000);
  const [activeMetricTab, setActiveMetricTab] = useState("median");

  // Filtered roles based on category pill
  const filteredRoles = useMemo(() => {
    if (selectedRoleFilter === "All") return TOP_ROLES_DATA;
    return TOP_ROLES_DATA.filter((r) => r.category === selectedRoleFilter);
  }, [selectedRoleFilter]);

  // Dynamic percentile computation for benchmark widget
  const benchmarkStats = useMemo(() => {
    const salary = Number(userBenchmarkSalary) || 0;
    let percentile;
    if (salary < 30000) percentile = Math.max(1, Math.round((salary / 30000) * 5));
    else if (salary < 50602) percentile = Math.round(5 + ((salary - 30000) / 20602) * 5);
    else if (salary < 84975) percentile = Math.round(10 + ((salary - 50602) / 34373) * 15);
    else if (salary < 130000) percentile = Math.round(25 + ((salary - 84975) / 45025) * 25);
    else if (salary < 175000) percentile = Math.round(50 + ((salary - 130000) / 45000) * 25);
    else if (salary < 220000) percentile = Math.round(75 + ((salary - 175000) / 45000) * 15);
    else if (salary < 250000) percentile = Math.round(90 + ((salary - 220000) / 30000) * 5);
    else percentile = Math.min(99, Math.round(95 + ((salary - 250000) / 150000) * 4));

    const diffFromMedian = salary - 130000;
    const diffPct = ((diffFromMedian / 130000) * 100).toFixed(1);

    return {
      percentile,
      diffFromMedian,
      diffPct,
      isAbove: diffFromMedian >= 0,
    };
  }, [userBenchmarkSalary]);

  return (
    <div className="analytics-page">
      {/* --- Header Section --- */}
      <section className="analytics-hero">
        <div>
          <div className="analytics-eyebrow">
            <Sparkles size={15} />
            <span>SALARYIQ INTELLIGENCE PLATFORM</span>
          </div>
          <h1 className="analytics-title">Tech Compensation Analytics</h1>
          <p className="analytics-subtitle">
            Empirical compensation benchmarks, role valuations, and predictive feature
            drivers synthesized from <strong>2,584 verified records</strong> across the global data & AI market.
          </p>
        </div>

        <div className="analytics-quick-actions">
          {onNavigateToPredictor && (
            <button className="primary-predict-cta" onClick={onNavigateToPredictor}>
              <Sliders size={16} />
              <span>Test My Profile in Predictor</span>
              <ArrowUpRight size={16} />
            </button>
          )}
        </div>
      </section>

      {/* --- Top KPI Metric Cards --- */}
      <section className="analytics-kpi-grid">
        <div className="kpi-card">
          <div className="kpi-top">
            <span className="kpi-label">Market Median Salary</span>
            <div className="kpi-icon purple">
              <DollarSign size={18} />
            </div>
          </div>
          <div className="kpi-value">$130,000</div>
          <div className="kpi-footer">
            <span className="badge positive">
              <TrendingUp size={13} /> +18.4% YoY
            </span>
            <span className="kpi-sub">Mean: $133,409</span>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-top">
            <span className="kpi-label">Senior Tier Median</span>
            <div className="kpi-icon blue">
              <Award size={18} />
            </div>
          </div>
          <div className="kpi-value">$148,800</div>
          <div className="kpi-footer">
            <span className="badge blue-badge">+140% vs Entry</span>
            <span className="kpi-sub">1,554 practitioners</span>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-top">
            <span className="kpi-label">Remote Share</span>
            <div className="kpi-icon green">
              <Laptop size={18} />
            </div>
          </div>
          <div className="kpi-value">63.6%</div>
          <div className="kpi-footer">
            <span className="badge green-badge">$135,000 Median</span>
            <span className="kpi-sub">Fully remote parity</span>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-top">
            <span className="kpi-label">US Geo Multiplier</span>
            <div className="kpi-icon orange">
              <Globe size={18} />
            </div>
          </div>
          <div className="kpi-value">1.89×</div>
          <div className="kpi-footer">
            <span className="badge orange-badge">$151k vs $80k</span>
            <span className="kpi-sub">US vs Non-US average</span>
          </div>
        </div>
      </section>

      {/* --- Main 2-Column Analytical Charts --- */}
      <div className="analytics-grid-two">
        {/* Left: Experience Tier Progression */}
        <div className="chart-panel">
          <div className="panel-header-with-actions">
            <div>
              <div className="card-badge">CAREER PROGRESSION</div>
              <h3>Salary Growth by Experience Level</h3>
              <p className="card-desc">
                Progression of compensation across entry, intermediate, senior, and executive tiers.
              </p>
            </div>
            <div className="toggle-pill-group">
              <button
                className={`toggle-pill ${activeMetricTab === "median" ? "active" : ""}`}
                onClick={() => setActiveMetricTab("median")}
              >
                Median
              </button>
              <button
                className={`toggle-pill ${activeMetricTab === "mean" ? "active" : ""}`}
                onClick={() => setActiveMetricTab("mean")}
              >
                Average
              </button>
            </div>
          </div>

          <div className="chart-container" style={{ height: 320 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={EXPERIENCE_DATA} margin={{ top: 20, right: 20, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis dataKey="tier" stroke="#94a3b8" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                <YAxis
                  stroke="#94a3b8"
                  tick={{ fill: "#94a3b8", fontSize: 12 }}
                  tickFormatter={(val) => `$${val / 1000}k`}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar
                  dataKey={activeMetricTab === "median" ? "median" : "mean"}
                  name={activeMetricTab === "median" ? "Median Salary" : "Average Salary"}
                  fill="url(#expBarGradient)"
                  radius={[8, 8, 0, 0]}
                  barSize={46}
                />
                <defs>
                  <linearGradient id="expBarGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.95} />
                    <stop offset="100%" stopColor="#4f46e5" stopOpacity={0.65} />
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="tier-breakdown-row">
            {EXPERIENCE_DATA.map((item, idx) => (
              <div key={idx} className="tier-pill">
                <span className="tier-name">{item.tier.split(" ")[0]}</span>
                <strong>{formatUSD(item.median)}</strong>
                <span className="tier-growth">{item.growth}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Compensation Trajectory Over Time */}
        <div className="chart-panel">
          <div className="panel-header-with-actions">
            <div>
              <div className="card-badge blue">MACRO TRENDS</div>
              <h3>Historical Salary Trajectory (2020 – 2023)</h3>
              <p className="card-desc">
                Tracking post-pandemic demand surging and overall tech salary expansion.
              </p>
            </div>
          </div>

          <div className="chart-container" style={{ height: 320 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={HISTORICAL_TREND_DATA} margin={{ top: 20, right: 20, left: 10, bottom: 10 }}>
                <defs>
                  <linearGradient id="seniorArea" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.0} />
                  </linearGradient>
                  <linearGradient id="overallArea" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis dataKey="year" stroke="#94a3b8" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                <YAxis
                  stroke="#94a3b8"
                  tick={{ fill: "#94a3b8", fontSize: 12 }}
                  tickFormatter={(val) => `$${val / 1000}k`}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend
                  wrapperStyle={{ paddingTop: 12 }}
                  iconType="circle"
                  formatter={(value) => <span style={{ color: "#cbd5e1", fontSize: 12 }}>{value}</span>}
                />
                <Area
                  type="monotone"
                  dataKey="seniorSalary"
                  name="Senior Track ($)"
                  stroke="#3b82f6"
                  strokeWidth={2.5}
                  fillOpacity={1}
                  fill="url(#seniorArea)"
                />
                <Area
                  type="monotone"
                  dataKey="avgSalary"
                  name="Market Average ($)"
                  stroke="#8b5cf6"
                  strokeWidth={2.5}
                  fillOpacity={1}
                  fill="url(#overallArea)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="stat-callout-banner">
            <CheckCircle2 size={16} className="callout-icon" />
            <span>
              Average tech salaries rose by <strong>+49.0%</strong> from $92k in 2020 to $137k in 2023, led by high-demand AI & data infrastructure roles.
            </span>
          </div>
        </div>
      </div>

      {/* --- Top Paying Roles & Filter Bar --- */}
      <section className="chart-panel full-width-panel">
        <div className="panel-header-with-actions">
          <div>
            <div className="card-badge green">ROLE BENCHMARKS</div>
            <h3>Top Compensated Tech & Data Roles</h3>
            <p className="card-desc">
              Compare median and mean base compensations across specialized domain tracks.
            </p>
          </div>

          <div className="filter-button-group">
            <span className="filter-icon-label">
              <Filter size={14} /> Filter:
            </span>
            {["All", "Engineering", "Science", "Leadership", "Analytics"].map((cat) => (
              <button
                key={cat}
                className={`filter-btn ${selectedRoleFilter === cat ? "active" : ""}`}
                onClick={() => setSelectedRoleFilter(cat)}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        <div className="chart-container" style={{ height: 350 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={filteredRoles}
              layout="vertical"
              margin={{ top: 10, right: 30, left: 140, bottom: 10 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
              <XAxis
                type="number"
                stroke="#94a3b8"
                tick={{ fill: "#94a3b8", fontSize: 12 }}
                tickFormatter={(val) => `$${val / 1000}k`}
              />
              <YAxis
                type="category"
                dataKey="role"
                stroke="#94a3b8"
                tick={{ fill: "#cbd5e1", fontSize: 12 }}
                width={130}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="median" name="Median Salary" fill="#3b82f6" radius={[0, 6, 6, 0]} barSize={20}>
                {filteredRoles.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.category === "Leadership" ? "#ec4899" : entry.category === "Science" ? "#8b5cf6" : "#3b82f6"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* --- 2-Column: Company Size & Remote Work Breakdown --- */}
      <div className="analytics-grid-two">
        {/* Company Scale Impact */}
        <div className="chart-panel">
          <div className="panel-header-with-actions">
            <div>
              <div className="card-badge orange">ORGANIZATIONAL SCALE</div>
              <h3>Company Size vs. Compensation</h3>
              <p className="card-desc">Median salary compared by organizational employee tiers.</p>
            </div>
            <Building2 size={20} className="header-deco-icon" />
          </div>

          <div className="company-size-grid">
            {COMPANY_SIZE_DATA.map((item, idx) => (
              <div key={idx} className="size-metric-box">
                <div className="size-header">
                  <span className="size-label">{item.size}</span>
                  <span className="size-badge" style={{ backgroundColor: `${item.color}20`, color: item.color }}>
                    {item.label}
                  </span>
                </div>
                <div className="size-salary">{formatUSD(item.median)}</div>
                <div className="size-meta">
                  <span>Market Share: <strong>{item.share}</strong></span>
                  <span>Mean: {formatUSD(item.mean)}</span>
                </div>
                <div className="size-bar-bg">
                  <div
                    className="size-bar-fill"
                    style={{ width: `${(item.median / 150000) * 100}%`, backgroundColor: item.color }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="stat-insight-text">
            💡 <strong>Key takeaway:</strong> Mid-sized firms (50–250 employees) offer the highest median compensation ($135,000), driven by aggressive Series B/C tech startups competing against big tech for high-caliber talent.
          </div>
        </div>

        {/* Remote Work Distribution */}
        <div className="chart-panel">
          <div className="panel-header-with-actions">
            <div>
              <div className="card-badge green">WORK ARRANGEMENT</div>
              <h3>Remote Work Share & Pay</h3>
              <p className="card-desc">Distribution of roles and salary by workplace setting.</p>
            </div>
            <Laptop size={20} className="header-deco-icon" />
          </div>

          <div className="remote-container">
            <div className="pie-wrapper" style={{ width: 220, height: 220 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={REMOTE_DISTRIBUTION}
                    cx="50%"
                    cy="50%"
                    innerRadius={65}
                    outerRadius={95}
                    paddingAngle={4}
                    dataKey="value"
                  >
                    {REMOTE_DISTRIBUTION.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
              <div className="pie-center-label">
                <strong>63.6%</strong>
                <span>Remote</span>
              </div>
            </div>

            <div className="remote-legend-column">
              {REMOTE_DISTRIBUTION.map((item, idx) => (
                <div key={idx} className="remote-legend-item">
                  <div className="legend-indicator-line" style={{ backgroundColor: item.color }} />
                  <div>
                    <div className="legend-title">{item.name}</div>
                    <div className="legend-stats">
                      <strong>{formatUSD(item.median)}</strong>
                      <span>• {item.percentage}% of roles ({item.value} jobs)</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="stat-insight-text">
            💡 <strong>Finding:</strong> Fully remote roles demonstrate salary parity with strictly on-site positions ($135,000 vs $130,000), disproving the assumption that remote work carries a location discount.
          </div>
        </div>
      </div>

      {/* --- FEATURE IMPORTANCE & EXPLAINABILITY (Random Forest Actual Model) --- */}
      <FeatureImportance />

      {/* --- Interactive Percentile Benchmarking Tool --- */}
      <section className="chart-panel full-width-panel benchmark-panel">
        <div className="panel-header-with-actions">
          <div>
            <div className="card-badge">BENCHMARK TOOL</div>
            <h3>Where Do You Stand?</h3>
            <p className="card-desc">
              Input your target or current salary to benchmark against the global dataset.
            </p>
          </div>
        </div>

        <div className="benchmark-interactive-box">
          <div className="salary-slider-card">
            <label htmlFor="user-salary-input">Target / Current Salary (USD):</label>
            <div className="salary-input-wrapper">
              <span className="currency-prefix">$</span>
              <input
                id="user-salary-input"
                type="number"
                min="20000"
                max="500000"
                step="5000"
                value={userBenchmarkSalary}
                onChange={(e) => setUserBenchmarkSalary(e.target.value)}
              />
            </div>

            <input
              type="range"
              className="salary-range-slider"
              min="30000"
              max="350000"
              step="5000"
              value={userBenchmarkSalary}
              onChange={(e) => setUserBenchmarkSalary(e.target.value)}
            />
          </div>

          <div className="benchmark-result-callout">
            <div className="percentile-big-badge">
              <span className="percentile-number">{benchmarkStats.percentile}th</span>
              <span className="percentile-label">PERCENTILE</span>
            </div>
            <div className="percentile-text-wrap">
              <h4>
                You earn more than <strong>{benchmarkStats.percentile}%</strong> of data professionals.
              </h4>
              <p>
                {benchmarkStats.isAbove ? (
                  <span className="positive-diff">
                    <strong>+${Math.abs(benchmarkStats.diffFromMedian).toLocaleString()}</strong> (+{benchmarkStats.diffPct}%) above the market median ($130,000).
                  </span>
                ) : (
                  <span className="negative-diff">
                    <strong>-${Math.abs(benchmarkStats.diffFromMedian).toLocaleString()}</strong> ({benchmarkStats.diffPct}%) below the market median ($130,000).
                  </span>
                )}
              </p>
            </div>
          </div>

          <div className="percentile-ladder">
            <span className="ladder-title">MARKET PERCENTILE SCALE</span>
            <div className="ladder-points">
              {PERCENTILE_MARKS.map((mark, idx) => (
                <div
                  key={idx}
                  className={`ladder-point ${
                    Number(userBenchmarkSalary) >= mark.value ? "achieved" : ""
                  }`}
                >
                  <span className="point-badge">{mark.percentile}</span>
                  <span className="point-val">{formatUSD(mark.value)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
