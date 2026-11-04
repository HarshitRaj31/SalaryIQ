import { useState, lazy, Suspense } from "react";
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
  Star,
  Award,
} from "lucide-react";
import ErrorBoundary from "./components/ErrorBoundary";
import {
  AnalyticsSkeleton,
  FeatureImportanceSkeleton,
  PredictionHistorySkeleton,
  ShapDashboardSkeleton,
  ShapExplanationSkeleton,
} from "./components/Skeletons";
import "./App.css";

// Advanced Lazy Loading: non-critical & chart-driven views are code-split
const Analytics = lazy(() => import("./pages/Analytics"));
const FeatureImportance = lazy(() => import("./pages/FeatureImportance"));
const PredictionHistory = lazy(() => import("./pages/PredictionHistory"));
const ShapDashboard = lazy(() => import("./pages/ShapDashboard"));
const ShapExplanationView = lazy(() => import("./components/ShapExplanationView"));

const API_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const initialIndiaForm = {
  country: "IN",
  role_standard: "Data Scientist",
  experience_level: "SE",
  employment_type: "FT",
  primary_city_clean: "Bangalore",
  company_size: "L",
  remote_ratio: 50,
  experience_avg_yrs: 6.0,
  company_rating: 4.2,
  skills_count: 6,
  work_year: 2026,
};

const initialUsForm = {
  country: "US",
  job_title: "Data Scientist",
  experience_level: "SE",
  employment_type: "FT",
  company_location: "US",
  employee_residence: "US",
  company_size: "M",
  remote_ratio: 100,
  work_year: 2023,
};

const INDIA_ROLES = [
  "Data Scientist",
  "Data Analyst",
  "Data Engineer",
  "Machine Learning Engineer",
  "Business Analyst",
  "Python Developer",
];

const INDIA_CITIES = [
  "Bangalore",
  "Hyderabad",
  "Pune",
  "Mumbai",
  "Delhi / NCR",
  "Chennai",
  "Kolkata",
  "Ahmedabad",
  "Remote",
];

function App() {
  const [selectedCountry, setSelectedCountry] = useState("IN"); // "IN" | "US"
  const [indiaForm, setIndiaForm] = useState(initialIndiaForm);
  const [usForm, setUsForm] = useState(initialUsForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [mobileMenu, setMobileMenu] = useState(false);
  const [activeTab, setActiveTab] = useState("overview"); // "overview" | "predictor" | "analytics"
  const [showXaiExplanation, setShowXaiExplanation] = useState(false);
  const [shapLoading, setShapLoading] = useState(false);
  const [pdfLoading, setPdfLoading] = useState(false);

  const handleDownloadPdf = async () => {
    if (pdfLoading || !result) return;
    try {
      setPdfLoading(true);
      const { generatePdfReport } = await import("./utils/pdfReportGenerator");
      generatePdfReport({ ...result, ...activeForm });
    } catch (err) {
      console.error("Failed to load PDF report generator:", err);
    } finally {
      setPdfLoading(false);
    }
  };

  const handleToggleShapExplanation = async () => {
    if (showXaiExplanation) {
      setShowXaiExplanation(false);
      return;
    }

    setShowXaiExplanation(true);

    if (!result?.shap_explanation && !shapLoading) {
      try {
        setShapLoading(true);
        const response = await fetch(`${API_URL}/explain`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            ...activeForm,
            country: selectedCountry,
          }),
        });

        if (response.ok) {
          const shapData = await response.json();
          setResult((prev) => (prev ? { ...prev, shap_explanation: shapData } : prev));
        }
      } catch (err) {
        console.warn("Could not fetch real-time SHAP explanation, fallback baseline active:", err);
      } finally {
        setShapLoading(false);
      }
    }
  };

  const activeForm = selectedCountry === "IN" ? indiaForm : usForm;

  const handleCountrySwitch = (countryCode) => {
    if (countryCode === selectedCountry) return;
    setSelectedCountry(countryCode);
    setError("");
    setResult(null);
    setShowXaiExplanation(false);
  };

  const updateField = (event) => {
    const { name, value } = event.target;
    const numericFields = [
      "work_year",
      "remote_ratio",
      "experience_avg_yrs",
      "company_rating",
      "skills_count",
    ];

    const parsedVal = numericFields.includes(name)
      ? Number(value)
      : value;

    if (selectedCountry === "IN") {
      setIndiaForm((prev) => ({
        ...prev,
        [name]: parsedVal,
      }));
    } else {
      setUsForm((prev) => ({
        ...prev,
        [name]: parsedVal,
      }));
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    const payload = {
      ...activeForm,
      country: selectedCountry,
    };

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
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
              <p>Explore dual-country models (IN & US).</p>
            </div>
          </div>

          <div className="profile">
            <div className="avatar">S</div>
            <div className="profile-info">
              <strong>SalaryIQ User</strong>
              <span>Global workspace</span>
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
              Dual ML Engines Active
            </div>
            <div className="top-avatar">S</div>
          </div>
        </header>

        <div className="page-container">
          {activeTab === "shap-dashboard" ? (
            <ErrorBoundary componentName="SHAP Explainability Dashboard">
              <Suspense fallback={<ShapDashboardSkeleton />}>
                <ShapDashboard onNavigateToPredictor={() => setActiveTab("predictor")} />
              </Suspense>
            </ErrorBoundary>
          ) : activeTab === "history" ? (
            <ErrorBoundary componentName="Prediction History">
              <Suspense fallback={<PredictionHistorySkeleton />}>
                <PredictionHistory
                  onSelectProfile={(profile) => {
                    if (profile.country === "IN") {
                      setSelectedCountry("IN");
                      setIndiaForm((prev) => ({ ...prev, ...profile }));
                    } else {
                      setSelectedCountry("US");
                      setUsForm((prev) => ({ ...prev, ...profile }));
                    }
                    setActiveTab("predictor");
                  }}
                  onNavigateToPredictor={() => setActiveTab("predictor")}
                />
              </Suspense>
            </ErrorBoundary>
          ) : activeTab === "feature-importance" ? (
            <ErrorBoundary componentName="Feature Importance">
              <Suspense fallback={<FeatureImportanceSkeleton />}>
                <FeatureImportance standalone={true} />
              </Suspense>
            </ErrorBoundary>
          ) : activeTab === "analytics" ? (
            <ErrorBoundary componentName="Analytics Dashboard">
              <Suspense fallback={<AnalyticsSkeleton />}>
                <Analytics onNavigateToPredictor={() => setActiveTab("predictor")} />
              </Suspense>
            </ErrorBoundary>
          ) : (
            <>
              {activeTab === "overview" && (
                <>
                  <section id="overview" className="welcome-section">
                    <div>
                      <div className="eyebrow">
                        <Sparkles size={15} />
                        DUAL-COUNTRY SALARY INTELLIGENCE
                      </div>
                      <h1>Know your worth.</h1>
                      <p>
                        Estimate your earning potential with dedicated machine learning
                        models trained on verified compensation data for India and the United States.
                      </p>
                    </div>

                    <div className="date-chip">
                      <Clock3 size={16} />
                      Dual-Country ML System
                    </div>
                  </section>

                  <section className="stats-grid">
                    <div className="stat-card" onClick={() => setActiveTab("feature-importance")} style={{ cursor: "pointer" }}>
                      <div className="stat-top">
                        <span>India Model (INR / LPA)</span>
                        <div className="stat-icon orange">
                          <Activity size={19} />
                        </div>
                      </div>
                      <h2>Gradient Boosting</h2>
                      <div className="stat-foot">
                        <span className="positive">
                          <CheckCircle2 size={14} /> MAE: 5.27 LPA
                        </span>
                        <span>2,269 Clean Records</span>
                      </div>
                    </div>

                    <div className="stat-card" onClick={() => setActiveTab("feature-importance")} style={{ cursor: "pointer" }}>
                      <div className="stat-top">
                        <span>US Model (USD / Year)</span>
                        <div className="stat-icon purple">
                          <Cpu size={19} />
                        </div>
                      </div>
                      <h2>Random Forest</h2>
                      <div className="stat-foot">
                        <span className="positive">
                          <CheckCircle2 size={14} /> MAE: $39,039
                        </span>
                        <span>2,584 US Records</span>
                      </div>
                    </div>

                    <div className="stat-card" onClick={() => setActiveTab("analytics")} style={{ cursor: "pointer" }}>
                      <div className="stat-top">
                        <span>India Model R²</span>
                        <div className="stat-icon green">
                          <ChartNoAxesCombined size={19} />
                        </div>
                      </div>
                      <h2>48.87%</h2>
                      <div className="stat-foot">
                        <span>Test set R²</span>
                        <span>Top Indian Tech Roles</span>
                      </div>
                    </div>

                    <div className="stat-card" onClick={() => setActiveTab("history")} style={{ cursor: "pointer" }}>
                      <div className="stat-top">
                        <span>Coverage</span>
                        <div className="stat-icon blue">
                          <Users size={19} />
                        </div>
                      </div>
                      <h2>IN 🇮🇳 & US 🇺🇸</h2>
                      <div className="stat-foot">
                        <span>Dual Currency Support</span>
                        <span>INR & USD</span>
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
                      <p>Select your market country and fill in job details for ML estimation.</p>
                    </div>
                    <div className="heading-icon">
                      <BriefcaseBusiness size={21} />
                    </div>
                  </div>

                  {/* ---------------------------------------------------- */}
                  {/* 1. COUNTRY SELECTION TOGGLE */}
                  {/* ---------------------------------------------------- */}
                  <div className="country-selector-wrap">
                    <span className="country-selector-label">SELECT TARGET MARKET</span>
                    <div className="country-toggle-grid">
                      <button
                        type="button"
                        className={`country-toggle-btn ${selectedCountry === "IN" ? "active" : ""}`}
                        onClick={() => handleCountrySwitch("IN")}
                      >
                        <span className="country-flag">🇮🇳</span>
                        <div className="country-btn-text">
                          <span className="country-btn-title">India</span>
                          <span className="country-btn-sub">INR (₹) • LPA • Gradient Boosting</span>
                        </div>
                      </button>

                      <button
                        type="button"
                        className={`country-toggle-btn ${selectedCountry === "US" ? "active" : ""}`}
                        onClick={() => handleCountrySwitch("US")}
                      >
                        <span className="country-flag">🇺🇸</span>
                        <div className="country-btn-text">
                          <span className="country-btn-title">United States</span>
                          <span className="country-btn-sub">USD ($) • Annual • Random Forest</span>
                        </div>
                      </button>
                    </div>
                  </div>

                  <form onSubmit={handleSubmit}>
                    {/* ------------------------------------------------ */}
                    {/* 2. CONDITIONAL FORM: INDIA */}
                    {/* ------------------------------------------------ */}
                    {selectedCountry === "IN" ? (
                      <>
                        <div className="form-section-title">ROLE & EXPERIENCE (INDIA)</div>
                        <div className="form-grid">
                          <div className="form-group full-width">
                            <label htmlFor="role_standard">Job role / track</label>
                            <select
                              id="role_standard"
                              name="role_standard"
                              value={indiaForm.role_standard}
                              onChange={updateField}
                              required
                            >
                              {INDIA_ROLES.map((r) => (
                                <option key={r} value={r}>
                                  {r}
                                </option>
                              ))}
                            </select>
                          </div>

                          <div className="form-group">
                            <label htmlFor="experience_level">Seniority tier</label>
                            <select
                              id="experience_level"
                              name="experience_level"
                              value={indiaForm.experience_level}
                              onChange={updateField}
                              required
                            >
                              <option value="EN">Entry-level (EN)</option>
                              <option value="MI">Mid-level (MI)</option>
                              <option value="SE">Senior-level (SE)</option>
                              <option value="EX">Executive (EX)</option>
                            </select>
                          </div>

                          <div className="form-group">
                            <label htmlFor="experience_avg_yrs">
                              Years of experience
                            </label>
                            <input
                              id="experience_avg_yrs"
                              name="experience_avg_yrs"
                              type="number"
                              min="0"
                              max="30"
                              step="0.5"
                              value={indiaForm.experience_avg_yrs}
                              onChange={updateField}
                              required
                            />
                          </div>
                        </div>

                        <div className="form-section-title second-section">
                          LOCATION & EMPLOYER
                        </div>
                        <div className="form-grid">
                          <div className="form-group">
                            <label htmlFor="primary_city_clean">
                              <MapPin size={14} /> Primary tech city
                            </label>
                            <select
                              id="primary_city_clean"
                              name="primary_city_clean"
                              value={indiaForm.primary_city_clean}
                              onChange={updateField}
                              required
                            >
                              {INDIA_CITIES.map((c) => (
                                <option key={c} value={c}>
                                  {c}
                                </option>
                              ))}
                            </select>
                          </div>

                          <div className="form-group">
                            <label htmlFor="company_size">Company scale</label>
                            <select
                              id="company_size"
                              name="company_size"
                              value={indiaForm.company_size}
                              onChange={updateField}
                              required
                            >
                              <option value="S">Small (&lt; 50 employees)</option>
                              <option value="M">Medium (50 - 250 employees)</option>
                              <option value="L">Large (&gt; 250 employees)</option>
                            </select>
                          </div>

                          <div className="form-group">
                            <label htmlFor="company_rating">
                              <Star size={14} /> Company rating (1.0 - 5.0)
                            </label>
                            <input
                              id="company_rating"
                              name="company_rating"
                              type="number"
                              min="1.0"
                              max="5.0"
                              step="0.1"
                              value={indiaForm.company_rating}
                              onChange={updateField}
                              required
                            />
                          </div>

                          <div className="form-group">
                            <label htmlFor="skills_count">
                              <Award size={14} /> Number of verified skills
                            </label>
                            <input
                              id="skills_count"
                              name="skills_count"
                              type="number"
                              min="1"
                              max="25"
                              step="1"
                              value={indiaForm.skills_count}
                              onChange={updateField}
                              required
                            />
                          </div>
                        </div>

                        <div className="form-section-title second-section">
                          WORK ARRANGEMENT
                        </div>
                        <div className="form-grid">
                          <div className="form-group">
                            <label htmlFor="remote_ratio">
                              <Monitor size={14} /> Remote work setting
                            </label>
                            <select
                              id="remote_ratio"
                              name="remote_ratio"
                              value={indiaForm.remote_ratio}
                              onChange={updateField}
                            >
                              <option value={0}>On-site (0% remote)</option>
                              <option value={50}>Hybrid (50% remote)</option>
                              <option value={100}>Fully remote (100%)</option>
                            </select>
                          </div>

                          <div className="form-group">
                            <label htmlFor="employment_type">Employment type</label>
                            <select
                              id="employment_type"
                              name="employment_type"
                              value={indiaForm.employment_type}
                              onChange={updateField}
                              disabled
                            >
                              <option value="FT">Full-Time (FT)</option>
                            </select>
                          </div>
                        </div>
                      </>
                    ) : (
                      /* ------------------------------------------------ */
                      /* 3. CONDITIONAL FORM: UNITED STATES */
                      /* ------------------------------------------------ */
                      <>
                        <div className="form-section-title">JOB INFORMATION (US)</div>
                        <div className="form-grid">
                          <div className="form-group full-width">
                            <label htmlFor="job_title">Job title</label>
                            <input
                              id="job_title"
                              name="job_title"
                              value={usForm.job_title}
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
                              value={usForm.experience_level}
                              onChange={updateField}
                            >
                              <option value="EN">Entry-level (EN)</option>
                              <option value="MI">Mid-level (MI)</option>
                              <option value="SE">Senior-level (SE)</option>
                              <option value="EX">Executive (EX)</option>
                            </select>
                          </div>

                          <div className="form-group">
                            <label htmlFor="employment_type">Employment type</label>
                            <select
                              id="employment_type"
                              name="employment_type"
                              value={usForm.employment_type}
                              onChange={updateField}
                            >
                              <option value="FT">Full-time (FT)</option>
                              <option value="PT">Part-time (PT)</option>
                              <option value="CT">Contract (CT)</option>
                              <option value="FL">Freelance (FL)</option>
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
                              value={usForm.company_location}
                              onChange={updateField}
                              maxLength={2}
                              minLength={2}
                              placeholder="US"
                              required
                            />
                          </div>

                          <div className="form-group">
                            <label htmlFor="employee_residence">
                              <Globe2 size={14} /> Employee residence
                            </label>
                            <input
                              id="employee_residence"
                              name="employee_residence"
                              value={usForm.employee_residence}
                              onChange={updateField}
                              maxLength={2}
                              minLength={2}
                              placeholder="US"
                              required
                            />
                          </div>

                          <div className="form-group">
                            <label htmlFor="company_size">Company scale</label>
                            <select
                              id="company_size"
                              name="company_size"
                              value={usForm.company_size}
                              onChange={updateField}
                            >
                              <option value="S">Small (&lt; 50 employees)</option>
                              <option value="M">Medium (50 - 250 employees)</option>
                              <option value="L">Large (&gt; 250 employees)</option>
                            </select>
                          </div>

                          <div className="form-group">
                            <label htmlFor="work_year">Market year</label>
                            <input
                              id="work_year"
                              name="work_year"
                              type="number"
                              min="2020"
                              max="2035"
                              value={usForm.work_year}
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
                            value={usForm.remote_ratio}
                            onChange={updateField}
                          >
                            <option value={0}>On-site (0% remote)</option>
                            <option value={50}>Hybrid (50% remote)</option>
                            <option value={100}>Fully remote (100%)</option>
                          </select>
                        </div>
                      </>
                    )}

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
                          Running {selectedCountry === "IN" ? "India" : "US"} ML engine...
                        </>
                      ) : (
                        <>
                          <Sparkles size={18} />
                          Predict {selectedCountry === "IN" ? "India" : "US"} Salary
                          <ArrowUpRight size={18} />
                        </>
                      )}
                    </button>

                    <p className="form-disclaimer">
                      {selectedCountry === "IN"
                        ? "India predictions are generated via Gradient Boosting on 2,269 verified Indian tech records."
                        : "US predictions are generated via Random Forest on verified data science compensation records."}
                    </p>
                  </form>
                </div>

                {/* ---------------------------------------------------- */}
                {/* 4. RESULTS DISPLAY PANEL */}
                {/* ---------------------------------------------------- */}
                <div className="right-column">
                  <div className="result-panel">
                    <div className="result-heading">
                      <div>
                        <div className="section-eyebrow">
                          <Activity size={14} /> LIVE ML ESTIMATE
                        </div>
                        <h2>Predicted Compensation</h2>
                      </div>
                      <div className="result-icon">
                        <Wallet size={21} />
                      </div>
                    </div>

                    {result ? (
                      <div className="result-content">
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                          <span className="result-label">ESTIMATED COMPENSATION</span>
                          <span className={`country-badge ${result.country === "IN" ? "in" : "us"}`}>
                            {result.country === "IN" ? "🇮🇳 India (INR)" : "🇺🇸 United States (USD)"}
                          </span>
                        </div>

                        <div className="salary-value">
                          {result.formatted_salary || (result.country === "IN"
                            ? `₹${Number(result.predicted_salary).toFixed(2)} LPA`
                            : `$${Math.round(result.predicted_salary).toLocaleString()} / year`)}
                        </div>

                        <span className="salary-period">
                          {result.country === "IN"
                            ? "Lakhs Per Annum (INR) • Annual Cost-to-Company (CTC)"
                            : "USD / year • Base & Total Compensation"}
                        </span>

                        {/* Selected Profile Summary */}
                        <div className="profile-meta-grid">
                          <div className="profile-meta-item">
                            <span className="profile-meta-label">Role</span>
                            <span className="profile-meta-value">
                              {result.input_features?.role_standard || result.input_features?.job_title}
                            </span>
                          </div>
                          <div className="profile-meta-item">
                            <span className="profile-meta-label">Location</span>
                            <span className="profile-meta-value">
                              {result.input_features?.primary_city_clean || result.input_features?.company_location}
                            </span>
                          </div>
                          <div className="profile-meta-item">
                            <span className="profile-meta-label">Seniority Tier</span>
                            <span className="profile-meta-value">
                              {result.input_features?.experience_level}
                              {result.input_features?.experience_avg_yrs && ` (${result.input_features.experience_avg_yrs} yrs)`}
                            </span>
                          </div>
                          <div className="profile-meta-item">
                            <span className="profile-meta-label">Engine</span>
                            <span className="profile-meta-value" style={{ color: "#a78bfa" }}>
                              {result.model_used || (result.country === "IN" ? "Gradient Boosting" : "Random Forest")}
                            </span>
                          </div>
                        </div>

                        <div className="result-divider" />

                        <div className="range-title">
                          <span>MAE-based estimated range</span>
                          <span><CircleHelp size={14} /></span>
                        </div>

                        <div className="range-values">
                          <div>
                            <strong>
                              {result.salary_range?.formatted_min ||
                                (result.country === "IN"
                                  ? `₹${Number(result.salary_range?.min).toFixed(2)} LPA`
                                  : `$${Math.round(result.salary_range?.min).toLocaleString()}`)}
                            </strong>
                            <span>Lower estimate</span>
                          </div>
                          <div>
                            <strong>
                              {result.salary_range?.formatted_max ||
                                (result.country === "IN"
                                  ? `₹${Number(result.salary_range?.max).toFixed(2)} LPA`
                                  : `$${Math.round(result.salary_range?.max).toLocaleString()}`)}
                            </strong>
                            <span>Upper estimate</span>
                          </div>
                        </div>

                        <div className="range-track">
                          <div className="range-track-fill" />
                        </div>

                        <div className="margin-info">
                          <span>Model MAE margin</span>
                          <strong>
                            {result.country === "IN"
                              ? `₹${Number(result.confidence_margin_mae).toFixed(2)} LPA`
                              : `$${Math.round(result.confidence_margin_mae).toLocaleString()}`}
                          </strong>
                        </div>

                        <div className="success-note">
                          <CheckCircle2 size={16} />
                          Prediction generated using {result.model_used || (result.country === "IN" ? "Gradient Boosting" : "Random Forest")}
                        </div>

                        <button
                          type="button"
                          onClick={handleToggleShapExplanation}
                          disabled={shapLoading}
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
                            cursor: shapLoading ? "wait" : "pointer",
                            transition: "all 0.2s ease",
                            opacity: shapLoading ? 0.8 : 1,
                          }}
                        >
                          {shapLoading ? (
                            <>
                              <LoaderCircle size={14} className="spin" />
                              <span>Computing SHAP Attribution...</span>
                            </>
                          ) : (
                            <>
                              <Cpu size={14} />
                              <span>
                                {showXaiExplanation
                                  ? "Hide SHAP XAI Breakdown"
                                  : "Explain Prediction (SHAP XAI) 💡"}
                              </span>
                            </>
                          )}
                        </button>

                        {showXaiExplanation && (
                          <div style={{ marginTop: 14 }}>
                            <ErrorBoundary componentName="SHAP XAI Breakdown">
                              <Suspense fallback={<ShapExplanationSkeleton compact={true} />}>
                                <ShapExplanationView
                                  shapData={result.shap_explanation}
                                  loading={shapLoading}
                                  compact={true}
                                />
                              </Suspense>
                            </ErrorBoundary>
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
                            fontWeight: "650",
                            cursor: "pointer",
                          }}
                        >
                          <History size={14} />
                          View in Prediction History
                        </button>

                        <button
                          type="button"
                          onClick={handleDownloadPdf}
                          disabled={pdfLoading}
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
                            fontWeight: "650",
                            cursor: pdfLoading ? "wait" : "pointer",
                            opacity: pdfLoading ? 0.75 : 1,
                          }}
                        >
                          {pdfLoading ? (
                            <>
                              <LoaderCircle size={14} className="spin" />
                              <span>Generating PDF Report...</span>
                            </>
                          ) : (
                            <>
                              <FileText size={14} />
                              <span>Download PDF Report 📄</span>
                            </>
                          )}
                        </button>
                      </div>
                    ) : (
                      <div className="empty-result">
                        <div className="empty-illustration">
                          <Wallet size={34} />
                          <Sparkles size={17} className="empty-sparkle" />
                        </div>
                        <h3>
                          {selectedCountry === "IN"
                            ? "Indian Tech Compensation Intelligence"
                            : "US Tech Compensation Intelligence"}
                        </h3>
                        <p>
                          {selectedCountry === "IN"
                            ? "Fill in your role, experience, city, and skills to generate an INR (LPA) salary prediction."
                            : "Fill in your job title, experience level, and location to generate a USD annual salary estimate."}
                        </p>
                        <div className="empty-hint">
                          <ArrowDownRight size={16} />
                          Your {selectedCountry === "IN" ? "INR / LPA" : "USD / year"} result will appear here
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="insight-panel" id="analytics">
                    <div className="insight-title">
                      <div className="insight-icon">
                        <Activity size={18} />
                      </div>
                      <strong>Active Model Information</strong>
                    </div>
                    <p>
                      {selectedCountry === "IN"
                        ? "Indian tech salary prediction is powered by a tuned Gradient Boosting Regressor (5-fold CV RMSE: 7.80 LPA)."
                        : "US salary prediction is powered by a tuned Random Forest Regressor (5-fold CV RMSE: $50,578)."}
                    </p>
                    <div className="insight-row">
                      <span>Engine</span>
                      <strong>
                        {selectedCountry === "IN"
                          ? "Gradient Boosting Regressor"
                          : "Random Forest Regressor"}
                      </strong>
                    </div>
                    <div className="insight-row">
                      <span>Target Currency</span>
                      <strong>
                        {selectedCountry === "IN" ? "INR (₹) in Lakhs Per Annum" : "USD ($) Annual"}
                      </strong>
                    </div>
                    <div className="insight-row">
                      <span>Training Dataset</span>
                      <strong>
                        {selectedCountry === "IN"
                          ? "Indian Tech Jobs 2026 (2,269 Records)"
                          : "Global Data Science Salaries (2,584 Records)"}
                      </strong>
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
            <span>© 2026 SalaryIQ. Dual-country Machine Learning System.</span>
            <span>Powered by React & FastAPI</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;
