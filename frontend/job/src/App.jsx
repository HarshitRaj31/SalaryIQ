
import { useState } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  BriefcaseBusiness,
  ChartNoAxesCombined,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Clock3,
  Cpu,
  FileText,
  Globe2,
  History,
  LayoutDashboard,
  LoaderCircle,
  MapPin,
  Menu,
  Monitor,
  Sparkles,
  Users,
  Wallet,
  X,
} from "lucide-react";
import Analytics from "./pages/Analytics";
import FeatureImportance from "./pages/FeatureImportance";
import PredictionHistory from "./pages/PredictionHistory";
import ShapDashboard from "./pages/ShapDashboard";
import ShapExplanationView from "./components/ShapExplanationView";
import { generatePdfReport } from "./utils/pdfReportGenerator";
import "./App.css";

const API_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const initialForm = {
  work_year: 2023,
  experience_level: "SE",
  employment_type: "FT",
  job_title: "Data Scientist",
  employee_residence: "US",
  remote_ratio: 100,
  company_location: "US",
  company_size: "M",
};

function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [mobileMenu, setMobileMenu] = useState(false);
  const [activeTab, setActiveTab] = useState("overview"); // "overview" | "predictor" | "analytics"
  const [showXaiExplanation, setShowXaiExplanation] = useState(false);

  const updateField = (event) => {
    const { name, value } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]:
        name === "work_year" || name === "remote_ratio"
          ? Number(value)
          : value,
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(form),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Unable to generate salary prediction."
        );
      }

      setResult(data.prediction || data);
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the API. Make sure your FastAPI server is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (amount) =>
    new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    }).format(amount);

  return (
    <div className="app-shell">
      {mobileMenu && (
        <div
          className="mobile-overlay"
          onClick={() => setMobileMenu(false)}
        />
      )}

      <aside className={`sidebar ${mobileMenu ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-icon">
            <ChartNoAxesCombined size={23} />
          </div>
          <span>Salary<span className="brand-highlight">IQ</span></span>
          <button
            className="mobile-close"
            onClick={() => setMobileMenu(false)}
            aria-label="Close menu"
          >
            <X size={20} />
          </button>
        </div>

        <div className="workspace-label">WORKSPACE</div>

        <nav className="sidebar-nav">
          <button
            type="button"
            className={`nav-link ${activeTab === "overview" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("overview");
              setMobileMenu(false);
            }}
          >
            <LayoutDashboard size={19} />
            <span>Overview</span>
          </button>

          <button
            type="button"
            className={`nav-link ${activeTab === "predictor" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("predictor");
              setMobileMenu(false);
            }}
          >
            <Wallet size={19} />
            <span>Salary Predictor</span>
          </button>

          <button
            type="button"
            className={`nav-link ${activeTab === "analytics" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("analytics");
              setMobileMenu(false);
            }}
          >
            <ChartNoAxesCombined size={19} />
            <span>Analytics</span>
          </button>

          <button
            type="button"
            className={`nav-link ${activeTab === "feature-importance" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("feature-importance");
              setMobileMenu(false);
            }}
          >
            <Cpu size={19} />
            <span>Feature Importance</span>
          </button>

          <button
            type="button"
            className={`nav-link ${activeTab === "shap-dashboard" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("shap-dashboard");
              setMobileMenu(false);
            }}
          >
            <Sparkles size={19} />
            <span>SHAP Explainability</span>
          </button>

          <button
            type="button"
            className={`nav-link ${activeTab === "history" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("history");
              setMobileMenu(false);
            }}
          >
            <History size={19} />
            <span>Prediction History</span>
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-help">
            <div className="help-icon">
              <CircleHelp size={19} />
            </div>
            <div>
              <strong>Need help?</strong>
              <p>Explore how SalaryIQ works.</p>
            </div>
          </div>

          <div className="profile">
            <div className="avatar">S</div>
            <div className="profile-info">
              <strong>SalaryIQ User</strong>
              <span>Personal workspace</span>
            </div>
            <ChevronDown size={16} />
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <button
            className="menu-toggle"
            onClick={() => setMobileMenu(true)}
            aria-label="Open menu"
          >
            <Menu size={22} />
          </button>

          <div className="breadcrumb">
            Workspace <span>/</span>{" "}
            <strong>
              {activeTab === "analytics"
                ? "Analytics Dashboard"
                : activeTab === "predictor"
                ? "Salary Predictor"
                : activeTab === "shap-dashboard"
                ? "SHAP Explainability Dashboard"
                : activeTab === "feature-importance"
                ? "Model Feature Importance"
                : activeTab === "history"
                ? "Prediction History"
                : "Overview"}
            </strong>
          </div>

          <div className="topbar-right">
            <div className="api-status">
              <span className="status-dot" />
              ML Prediction
            </div>
            <div className="top-avatar">S</div>
          </div>
        </header>

        <div className="page-container">
          {activeTab === "shap-dashboard" ? (
            <ShapDashboard onNavigateToPredictor={() => setActiveTab("predictor")} />
          ) : activeTab === "history" ? (
            <PredictionHistory
              onSelectProfile={(profile) => {
                setForm((prev) => ({ ...prev, ...profile }));
              }}
              onNavigateToPredictor={() => setActiveTab("predictor")}
            />
          ) : activeTab === "feature-importance" ? (
            <FeatureImportance standalone={true} />
          ) : activeTab === "analytics" ? (
            <Analytics onNavigateToPredictor={() => setActiveTab("predictor")} />
          ) : (
            <>
              {activeTab === "overview" && (
                <>
                  <section id="overview" className="welcome-section">
                    <div>
                      <div className="eyebrow">
                        <Sparkles size={15} />
                        INTELLIGENT SALARY INSIGHTS
                      </div>
                      <h1>Know your worth.</h1>
                      <p>
                        Explore salary insights and estimate your earning potential
                        with machine learning.
                      </p>
                    </div>

                    <div className="date-chip">
                      <Clock3 size={16} />
                      ML-powered estimation
                    </div>
                  </section>

                  <section className="stats-grid">
                    <div className="stat-card" onClick={() => setActiveTab("feature-importance")} style={{ cursor: "pointer" }}>
                      <div className="stat-top">
                        <span>Prediction engine</span>
                        <div className="stat-icon purple">
                          <Activity size={19} />
                        </div>
                      </div>
                      <h2>Random Forest</h2>
                      <div className="stat-foot">
                        <span className="positive">
                          <CheckCircle2 size={14} /> Explainability
                        </span>
                        <span>View Feature Importance</span>
                      </div>
                    </div>

                    <div className="stat-card" onClick={() => setActiveTab("analytics")} style={{ cursor: "pointer" }}>
                      <div className="stat-top">
                        <span>Model R² score</span>
                        <div className="stat-icon blue">
                          <ChartNoAxesCombined size={19} />
                        </div>
                      </div>
                      <h2>39.57%</h2>
                      <div className="stat-foot">
                        <span>Test set</span>
                        <span>Random Forest</span>
                      </div>
                    </div>

                    <div className="stat-card" onClick={() => setActiveTab("analytics")} style={{ cursor: "pointer" }}>
                      <div className="stat-top">
                        <span>Mean absolute error</span>
                        <div className="stat-icon orange">
                          <Wallet size={19} />
                        </div>
                      </div>
                      <h2>$39,039</h2>
                      <div className="stat-foot">
                        <span>USD</span>
                        <span>Test set MAE</span>
                      </div>
                    </div>

                    <div className="stat-card" onClick={() => setActiveTab("analytics")} style={{ cursor: "pointer" }}>
                      <div className="stat-top">
                        <span>Training records</span>
                        <div className="stat-icon green">
                          <Users size={19} />
                        </div>
                      </div>
                      <h2>2,584</h2>
                      <div className="stat-foot">
                        <span>Processed dataset</span>
                        <span>Job salary records</span>
                      </div>
                    </div>
                  </section>
                </>
              )}

          <section className="content-grid">
            <div id="predictor" className="panel predictor-panel">
              <div className="panel-heading">
                <div>
                  <div className="section-eyebrow">
                    <Sparkles size={14} /> PREDICTION TOOL
                  </div>
                  <h2>Salary Predictor</h2>
                  <p>Enter job details to estimate annual compensation.</p>
                </div>
                <div className="heading-icon">
                  <BriefcaseBusiness size={21} />
                </div>
              </div>

              <form onSubmit={handleSubmit}>
                <div className="form-section-title">JOB INFORMATION</div>

                <div className="form-grid">
                  <div className="form-group full-width">
                    <label htmlFor="job_title">Job title</label>
                    <input
                      id="job_title"
                      name="job_title"
                      value={form.job_title}
                      onChange={updateField}
                      placeholder="e.g. Data Scientist"
                      required
                      minLength={2}
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="experience_level">Experience level</label>
                    <select
                      id="experience_level"
                      name="experience_level"
                      value={form.experience_level}
                      onChange={updateField}
                    >
                      <option value="EN">Entry-level</option>
                      <option value="MI">Mid-level</option>
                      <option value="SE">Senior-level</option>
                      <option value="EX">Executive</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label htmlFor="employment_type">Employment type</label>
                    <select
                      id="employment_type"
                      name="employment_type"
                      value={form.employment_type}
                      onChange={updateField}
                    >
                      <option value="FT">Full-time</option>
                      <option value="PT">Part-time</option>
                      <option value="CT">Contract</option>
                      <option value="FL">Freelance</option>
                    </select>
                  </div>
                </div>

                <div className="form-section-title second-section">
                  LOCATION & COMPANY
                </div>

                <div className="form-grid">
                  <div className="form-group">
                    <label htmlFor="company_location">
                      <MapPin size={14} /> Company location
                    </label>
                    <input
                      id="company_location"
                      name="company_location"
                      value={form.company_location}
                      onChange={updateField}
                      maxLength={2}
                      minLength={2}
                      placeholder="US"
                      required
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="employee_residence">
                      <Globe2 size={14} /> Your residence
                    </label>
                    <input
                      id="employee_residence"
                      name="employee_residence"
                      value={form.employee_residence}
                      onChange={updateField}
                      maxLength={2}
                      minLength={2}
                      placeholder="US"
                      required
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="company_size">Company size</label>
                    <select
                      id="company_size"
                      name="company_size"
                      value={form.company_size}
                      onChange={updateField}
                    >
                      <option value="S">Small</option>
                      <option value="M">Medium</option>
                      <option value="L">Large</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label htmlFor="work_year">Work year</label>
                    <input
                      id="work_year"
                      name="work_year"
                      type="number"
                      min="2020"
                      max="2035"
                      value={form.work_year}
                      onChange={updateField}
                      required
                    />
                  </div>
                </div>

                <div className="form-section-title second-section">
                  WORK ARRANGEMENT
                </div>

                <div className="form-group">
                  <label htmlFor="remote_ratio">
                    <Monitor size={14} /> Remote work arrangement
                  </label>
                  <select
                    id="remote_ratio"
                    name="remote_ratio"
                    value={form.remote_ratio}
                    onChange={updateField}
                  >
                    <option value={0}>On-site (0%)</option>
                    <option value={50}>Hybrid (50%)</option>
                    <option value={100}>Fully remote (100%)</option>
                  </select>
                </div>

                {error && (
                  <div className="error-message">
                    <CircleHelp size={17} />
                    {error}
                  </div>
                )}

                <button className="predict-button" type="submit" disabled={loading}>
                  {loading ? (
                    <>
                      <LoaderCircle className="spin" size={19} />
                      Analyzing your profile...
                    </>
                  ) : (
                    <>
                      <Sparkles size={18} />
                      Predict my salary
                      <ArrowUpRight size={18} />
                    </>
                  )}
                </button>

                <p className="form-disclaimer">
                  Predictions are estimates based on historical salary data.
                </p>
              </form>
            </div>

            <div className="right-column">
              <div className="result-panel">
                <div className="result-heading">
                  <div>
                    <div className="section-eyebrow">
                      <Activity size={14} /> LIVE RESULT
                    </div>
                    <h2>Your estimation</h2>
                  </div>
                  <div className="result-icon">
                    <Wallet size={21} />
                  </div>
                </div>

                {result ? (
                  <div className="result-content">
                    <span className="result-label">ESTIMATED ANNUAL SALARY</span>
                    <div className="salary-value">
                      {formatCurrency(result.predicted_salary_usd)}
                    </div>
                    <span className="salary-period">USD / year</span>

                    <div className="result-divider" />

                    <div className="range-title">
                      <span>MAE-based estimated range</span>
                      <span><CircleHelp size={14} /></span>
                    </div>

                    <div className="range-values">
                      <div>
                        <strong>{formatCurrency(result.salary_range.min)}</strong>
                        <span>Lower estimate</span>
                      </div>
                      <div>
                        <strong>{formatCurrency(result.salary_range.max)}</strong>
                        <span>Upper estimate</span>
                      </div>
                    </div>

                    <div className="range-track">
                      <div className="range-track-fill" />
                    </div>

                    <div className="margin-info">
                      <span>MAE margin</span>
                      <strong>{formatCurrency(result.confidence_margin_mae)}</strong>
                    </div>

                    <div className="success-note">
                      <CheckCircle2 size={16} />
                      Prediction generated successfully
                    </div>

                    <button
                      type="button"
                      onClick={() => setShowXaiExplanation((prev) => !prev)}
                      style={{
                        marginTop: 12,
                        width: "100%",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        gap: 6,
                        background: showXaiExplanation
                          ? "rgba(56, 189, 248, 0.15)"
                          : "rgba(15, 23, 42, 0.6)",
                        border: "1px solid #38bdf8",
                        color: "#38bdf8",
                        padding: "9px 12px",
                        borderRadius: "8px",
                        fontSize: "12.5px",
                        fontWeight: "700",
                        cursor: "pointer",
                        transition: "all 0.2s ease",
                      }}
                    >
                      <Cpu size={14} />
                      <span>
                        {showXaiExplanation
                          ? "Hide SHAP XAI Breakdown"
                          : "Explain Prediction (SHAP XAI) 💡"}
                      </span>
                    </button>

                    {showXaiExplanation && (
                      <div style={{ marginTop: 14 }}>
                        <ShapExplanationView
                          shapData={result.shap_explanation}
                          compact={true}
                        />
                      </div>
                    )}

                    <button
                      type="button"
                      onClick={() => setActiveTab("history")}
                      style={{
                        marginTop: 12,
                        width: "100%",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        gap: 6,
                        background: "#1e293b",
                        border: "1px solid #334155",
                        color: "#cbd5e1",
                        padding: "8px 12px",
                        borderRadius: "8px",
                        fontSize: "12.5px",
                        fontWeight: "600",
                        cursor: "pointer",
                      }}
                    >
                      <History size={14} />
                      View in Prediction History
                    </button>

                    <button
                      type="button"
                      onClick={() => generatePdfReport({ ...result, ...form })}
                      style={{
                        marginTop: 8,
                        width: "100%",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        gap: 6,
                        background: "linear-gradient(135deg, #7c3aed 0%, #6366f1 100%)",
                        border: "1px solid #7c3aed",
                        color: "#ffffff",
                        padding: "8px 12px",
                        borderRadius: "8px",
                        fontSize: "12.5px",
                        fontWeight: "600",
                        cursor: "pointer",
                      }}
                    >
                      <FileText size={14} />
                      Download PDF Report 📄
                    </button>
                  </div>
                ) : (
                  <div className="empty-result">
                    <div className="empty-illustration">
                      <Wallet size={34} />
                      <Sparkles size={17} className="empty-sparkle" />
                    </div>
                    <h3>Your next opportunity starts here.</h3>
                    <p>
                      Complete the job profile and generate a salary estimate
                      using your machine learning model.
                    </p>
                    <div className="empty-hint">
                      <ArrowDownRight size={16} />
                      Your result will appear here
                    </div>
                  </div>
                )}
              </div>

              <div className="insight-panel" id="analytics">
                <div className="insight-title">
                  <div className="insight-icon">
                    <Activity size={18} />
                  </div>
                  <strong>Model information</strong>
                </div>
                <p>
                  Your prediction is generated using the trained Random Forest
                  regression pipeline.
                </p>
                <div className="insight-row">
                  <span>Model</span>
                  <strong>Random Forest</strong>
                </div>
                <div className="insight-row">
                  <span>Currency</span>
                  <strong>USD</strong>
                </div>
                <div className="insight-row">
                  <span>Dataset</span>
                  <strong>Data Science Salaries</strong>
                </div>
              </div>
            </div>
          </section>

              {activeTab === "overview" && (
                <div
                  className="analytics-teaser-card"
                  onClick={() => setActiveTab("analytics")}
                >
                  <div className="teaser-left">
                    <div className="teaser-icon">
                      <ChartNoAxesCombined size={24} />
                    </div>
                    <div className="teaser-text">
                      <h3>Explore Full Market Compensation Analytics</h3>
                      <p>
                        Role benchmarks, experience tier premiums, remote work trends, and ML feature attribution.
                      </p>
                    </div>
                  </div>
                  <div className="teaser-btn">
                    <span>Open Analytics Dashboard</span>
                    <ArrowUpRight size={16} />
                  </div>
                </div>
              )}
            </>
          )}

          <footer className="footer">
            <span>© 2026 SalaryIQ. Built with machine learning.</span>
            <span>Powered by React & FastAPI</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;
