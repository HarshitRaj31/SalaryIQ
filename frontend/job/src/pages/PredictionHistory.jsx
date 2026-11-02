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
  Legend,
} from "recharts";
import {
  History,
  Database,
  Search,
  RefreshCw,
  Download,
  Trash2,
  Calendar,
  Sparkles,
  MapPin,
  Laptop,
  Building2,
  DollarSign,
  TrendingUp,
  Eye,
  X,
  FileText,
  Printer,
  AlertCircle,
  CheckCircle2,
  CheckSquare,
  Square,
  Scale,
  ArrowUpRight,
  Columns3,
  BarChart3,
  Layers,
} from "lucide-react";
import {
  generatePdfReport,
  generateComparisonPdf,
} from "../utils/pdfReportGenerator";
import ShapExplanationView from "../components/ShapExplanationView";
import "./PredictionHistory.css";

const API_URL = "http://127.0.0.1:8000";

const experienceLabels = {
  EN: "Entry-level (EN)",
  MI: "Mid-level (MI)",
  SE: "Senior-level (SE)",
  EX: "Executive (EX)",
};

const EXP_COLORS = {
  EN: "#34d399",
  MI: "#60a5fa",
  SE: "#c084fc",
  EX: "#fbbf24",
};

const EXP_SORT_ORDER = {
  EN: 1,
  MI: 2,
  SE: 3,
  EX: 4,
};

const formatMoney = (val) => {
  if (val === undefined || val === null || isNaN(val)) return "$0";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(val);
};

// Custom tooltip for Recharts in comparison modal
function CompareChartTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  const data = payload[0]?.payload;
  return (
    <div className="compare-chart-tooltip">
      <div className="tooltip-title">{data?.fullName || label}</div>
      {data?.experienceTier && (
        <div className="tooltip-sub">Tier: {data.experienceTier}</div>
      )}
      {payload.map((entry, idx) => (
        <div key={idx} className="tooltip-row">
          <span
            className="tooltip-dot"
            style={{ backgroundColor: entry.color || entry.fill }}
          />
          <span className="tooltip-label">{entry.name}:</span>
          <strong className="tooltip-value">{formatMoney(entry.value)}</strong>
        </div>
      ))}
    </div>
  );
}

function formatDate(isoString) {
  if (!isoString) return "N/A";
  try {
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return isoString;
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    }).format(date);
  } catch {
    return isoString;
  }
}

function getExperienceClass(level) {
  switch (level) {
    case "EN":
      return "exp-en";
    case "MI":
      return "exp-mi";
    case "SE":
      return "exp-se";
    case "EX":
      return "exp-ex";
    default:
      return "neutral";
  }
}

export default function PredictionHistory({
  onNavigateToPredictor,
}) {
  const [history, setHistory] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isLive, setIsLive] = useState(false);
  const [errorNotice, setErrorNotice] = useState(null);

  // Search, filter, and sorting states
  const [searchQuery, setSearchQuery] = useState("");
  const [experienceFilter, setExperienceFilter] = useState("ALL");
  const [remoteFilter, setRemoteFilter] = useState("ALL");
  const [sortBy, setSortBy] = useState("latest"); // "latest" | "salary-desc" | "salary-asc" | "role"

  // Comparison selection state (max 4 records)
  const [selectedRecordIds, setSelectedRecordIds] = useState([]);
  const [showCompareModal, setShowCompareModal] = useState(false);
  const [compareViewMode, setCompareViewMode] = useState("all"); // "all" | "grid" | "charts"

  // Modal details states
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [detailsError, setDetailsError] = useState("");
  const [detailTab, setDetailTab] = useState("overview"); // "overview" | "xai"

  // PDF Report Customizer states
  const [showPdfCustomizer, setShowPdfCustomizer] = useState(false);
  const [pdfOptions, setPdfOptions] = useState({
    candidateName: "Alex Morgan",
    targetCompany: "Enterprise AI Labs",
    preparedBy: "SalaryIQ Intelligence Engine",
    reportTitle: "Executive Compensation & Prediction Audit",
    customNotes: "Verified market compensation benchmark prepared for upcoming performance and compensation review.",
    includeConfidence: true,
    includeNegotiation: true,
    includeExplainability: true,
    themeColor: "purple",
  });

  const [deletingId, setDeletingId] = useState(null);
  const [notification, setNotification] = useState(null);
  const [refreshCounter, setRefreshCounter] = useState(0);

  // Fetch data from SQLite backend
  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();

    const fetchHistoryData = async () => {
      try {
        // Fetch paginated history from SQLite backend
        let historyRes = await fetch(`${API_URL}/history?limit=100`, {
          signal: controller.signal,
        });

        // Prefix fallback if needed
        if (!historyRes.ok) {
          historyRes = await fetch(
            `${API_URL}/api/v1/prediction/history?limit=100`,
            { signal: controller.signal }
          );
        }

        if (!historyRes.ok) {
          throw new Error(`History endpoint returned HTTP ${historyRes.status}`);
        }

        const historyData = await historyRes.json();

        // Fetch analytical summary from SQLite
        let summaryRes = await fetch(`${API_URL}/history/summary`, {
          signal: controller.signal,
        });
        if (!summaryRes.ok) {
          summaryRes = await fetch(
            `${API_URL}/api/v1/prediction/history/summary`,
            { signal: controller.signal }
          );
        }

        const summaryData = summaryRes.ok ? await summaryRes.json() : null;

        if (isMounted) {
          const records = historyData.history || [];
          setHistory(records);
          if (summaryData) setSummary(summaryData);
          setIsLive(true);
          setErrorNotice(null);
        }
      } catch (err) {
        if (!isMounted || err.name === "AbortError") return;
        console.warn("Could not reach SQLite history API:", err);
        setIsLive(false);
        setErrorNotice(
          "FastAPI SQLite backend is currently offline. Start backend/app/main.py to view live records."
        );
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchHistoryData();

    return () => {
      isMounted = false;
      controller.abort();
    };
  }, [refreshCounter]);

  // Refresh handler
  const handleRefresh = useCallback(() => {
    setLoading(true);
    setRefreshCounter((prev) => prev + 1);
  }, []);

  // View details function
  const handleViewDetails = async (id) => {
    try {
      setDetailsLoading(true);
      setDetailsError("");
      setSelectedPrediction(null);
      setDetailTab("overview");

      const response = await fetch(`${API_URL}/history/${id}`);

      if (!response.ok) {
        throw new Error("Unable to load prediction details.");
      }

      const result = await response.json();

      // Ensure SHAP explanation is present for older records
      if (!result.shap_explanation) {
        try {
          const explainRes = await fetch(`${API_URL}/history/${id}/explain`);
          if (explainRes.ok) {
            result.shap_explanation = await explainRes.json();
          }
        } catch {
          // non-blocking fallback
        }
      }

      setSelectedPrediction(result);
    } catch (err) {
      setDetailsError(err.message);
    } finally {
      setDetailsLoading(false);
    }
  };

  // Helper notification toaster
  const showNotification = useCallback((message) => {
    setNotification(message);
    setTimeout(() => {
      setNotification(null);
    }, 3500);
  }, []);

  // Generate customized PDF report
  const handleGeneratePdf = useCallback(() => {
    if (!selectedPrediction) return;
    const ok = generatePdfReport(selectedPrediction, pdfOptions);
    if (ok) {
      setShowPdfCustomizer(false);
      showNotification("PDF report generated! Save or print directly in the opened dialog.");
    }
  }, [selectedPrediction, pdfOptions, showNotification]);

  // Delete single prediction record
  const handleDelete = async (id) => {
    if (!window.confirm(`Delete prediction record #${id} from SQLite database?`)) {
      return;
    }

    setDeletingId(id);
    try {
      let res = await fetch(`${API_URL}/history/${id}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        res = await fetch(`${API_URL}/api/v1/prediction/history/${id}`, {
          method: "DELETE",
        });
      }

      if (res.ok) {
        setHistory((prev) => prev.filter((item) => item.id !== id));
        setSelectedRecordIds((prev) => {
          const updated = prev.filter((recordId) => recordId !== id);
          if (updated.length < 2) {
            setShowCompareModal(false);
          }
          return updated;
        });
        if (selectedPrediction?.id === id) {
          setSelectedPrediction(null);
        }
        showNotification("Record deleted successfully from SQLite database.");
        handleRefresh();
      } else {
        throw new Error("Deletion failed on server.");
      }
    } catch (err) {
      alert(`Error deleting record: ${err.message}`);
    } finally {
      setDeletingId(null);
    }
  };

  // Clear all prediction history
  const handleClearAll = async () => {
    if (
      !window.confirm(
        "Are you sure you want to permanently clear all prediction history records in SQLite?"
      )
    ) {
      return;
    }

    setLoading(true);
    try {
      let res = await fetch(`${API_URL}/history`, {
        method: "DELETE",
      });
      if (!res.ok) {
        res = await fetch(`${API_URL}/api/v1/prediction/history`, {
          method: "DELETE",
        });
      }

      if (res.ok) {
        setHistory([]);
        setSummary(null);
        setSelectedRecordIds([]);
        setShowCompareModal(false);
        if (selectedPrediction) setSelectedPrediction(null);
        showNotification("All prediction records cleared from SQLite database.");
        handleRefresh();
      } else {
        throw new Error("Clear history failed on server.");
      }
    } catch (err) {
      alert(`Error clearing history: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Export history to CSV file
  const handleExportCSV = useCallback(() => {
    if (!history.length) {
      alert("No prediction records available to export.");
      return;
    }

    const headers = [
      "ID",
      "Timestamp",
      "Job Title",
      "Experience Level",
      "Employment Type",
      "Employee Residence",
      "Company Location",
      "Company Size",
      "Remote Ratio (%)",
      "Work Year",
      "Predicted Salary (USD)",
      "Range Min (USD)",
      "Range Max (USD)",
      "Confidence Margin MAE (USD)",
      "Model Version",
    ];

    const rows = history.map((item) => [
      item.id,
      `"${item.timestamp}"`,
      `"${item.job_title}"`,
      `"${item.experience_level}"`,
      `"${item.employment_type}"`,
      `"${item.employee_residence}"`,
      `"${item.company_location}"`,
      `"${item.company_size}"`,
      item.remote_ratio,
      item.work_year,
      item.predicted_salary_usd,
      item.salary_range?.min || "",
      item.salary_range?.max || "",
      item.confidence_margin_mae || "",
      `"${item.model_version}"`,
    ]);

    const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join(
      "\n"
    );
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", "salaryiq_predictions_export.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showNotification("CSV export downloaded successfully.");
  }, [history, showNotification]);

  // Filtered and sorted records
  const filteredHistory = useMemo(() => {
    return history
      .filter((item) => {
        // Search filter
        if (searchQuery.trim()) {
          const q = searchQuery.toLowerCase();
          const matchTitle = item.job_title?.toLowerCase().includes(q);
          const matchComp = item.company_location?.toLowerCase().includes(q);
          const matchRes = item.employee_residence?.toLowerCase().includes(q);
          if (!matchTitle && !matchComp && !matchRes) return false;
        }

        // Experience filter
        if (
          experienceFilter !== "ALL" &&
          item.experience_level !== experienceFilter
        ) {
          return false;
        }

        // Remote filter
        if (
          remoteFilter !== "ALL" &&
          String(item.remote_ratio) !== String(remoteFilter)
        ) {
          return false;
        }

        return true;
      })
      .sort((a, b) => {
        if (sortBy === "salary-desc") {
          return b.predicted_salary_usd - a.predicted_salary_usd;
        }
        if (sortBy === "salary-asc") {
          return a.predicted_salary_usd - b.predicted_salary_usd;
        }
        if (sortBy === "role") {
          return (a.job_title || "").localeCompare(b.job_title || "");
        }
        // "latest" default
        return b.id - a.id;
      });
  }, [history, searchQuery, experienceFilter, remoteFilter, sortBy]);

  // Selection handlers for side-by-side comparison (max 4)
  const toggleSelectRecord = useCallback(
    (id, e) => {
      if (e) e.stopPropagation();
      setSelectedRecordIds((prev) => {
        if (prev.includes(id)) {
          return prev.filter((itemId) => itemId !== id);
        }
        if (prev.length >= 4) {
          showNotification("Maximum 4 predictions can be selected for comparison.");
          return prev;
        }
        return [...prev, id];
      });
    },
    [showNotification]
  );

  const handleSelectAllVisible = useCallback(() => {
    const visibleIds = filteredHistory.map((item) => item.id);
    if (visibleIds.length === 0) return;

    const firstFour = visibleIds.slice(0, 4);
    const allSelected = firstFour.every((id) => selectedRecordIds.includes(id));
    if (allSelected) {
      setSelectedRecordIds([]);
      showNotification("Selection cleared.");
    } else {
      setSelectedRecordIds(firstFour);
      showNotification(
        `Selected ${firstFour.length} record${
          firstFour.length > 1 ? "s" : ""
        } for comparison.`
      );
    }
  }, [filteredHistory, selectedRecordIds, showNotification]);

  const handleClearSelection = useCallback(() => {
    setSelectedRecordIds([]);
    setShowCompareModal(false);
  }, []);

  // Compared records list
  const selectedRecords = useMemo(() => {
    return history.filter((item) => selectedRecordIds.includes(item.id));
  }, [history, selectedRecordIds]);

  // Comparison metrics calculation
  const comparisonStats = useMemo(() => {
    if (selectedRecords.length < 2) return null;
    const salaries = selectedRecords.map((r) => r.predicted_salary_usd);
    const minSalary = Math.min(...salaries);
    const maxSalary = Math.max(...salaries);
    const diff = maxSalary - minSalary;
    const pctDiff = minSalary > 0 ? Math.round((diff / minSalary) * 100) : 0;
    const avgSalary = Math.round(
      salaries.reduce((sum, s) => sum + s, 0) / salaries.length
    );

    return {
      minSalary,
      maxSalary,
      diff,
      pctDiff,
      avgSalary,
      highestRecord: selectedRecords.find(
        (r) => r.predicted_salary_usd === maxSalary
      ),
      lowestRecord: selectedRecords.find(
        (r) => r.predicted_salary_usd === minSalary
      ),
    };
  }, [selectedRecords]);

  // Remove individual item from comparison modal
  const handleRemoveFromCompare = useCallback((id) => {
    setSelectedRecordIds((prev) => {
      const updated = prev.filter((item) => item !== id);
      if (updated.length < 2) {
        setShowCompareModal(false);
      }
      return updated;
    });
  }, []);

  // Chart 1: Direct Salary Comparison Data
  const salaryComparisonChartData = useMemo(() => {
    if (!comparisonStats) return [];
    return selectedRecords.map((item) => {
      const isHighest = item.id === comparisonStats.highestRecord?.id;
      const isLowest = item.id === comparisonStats.lowestRecord?.id;
      return {
        id: item.id,
        name:
          item.job_title.length > 15
            ? `${item.job_title.slice(0, 14)}...`
            : item.job_title,
        fullName: item.job_title,
        salary: item.predicted_salary_usd,
        formattedSalary:
          item.formatted_salary || formatMoney(item.predicted_salary_usd),
        experience: item.experience_level,
        experienceTier:
          experienceLabels[item.experience_level] || item.experience_level,
        isHighest,
        isLowest,
        fill: isHighest ? "#10b981" : isLowest ? "#f59e0b" : "#8b5cf6",
      };
    });
  }, [selectedRecords, comparisonStats]);

  // Chart 2: Experience-Level Comparison Data
  const experienceComparisonChartData = useMemo(() => {
    const sorted = [...selectedRecords].sort(
      (a, b) =>
        (EXP_SORT_ORDER[a.experience_level] || 99) -
        (EXP_SORT_ORDER[b.experience_level] || 99)
    );
    return sorted.map((item) => ({
      id: item.id,
      label: `${item.experience_level} • ${
        item.job_title.length > 12
          ? `${item.job_title.slice(0, 11)}...`
          : item.job_title
      }`,
      fullName: item.job_title,
      experienceTier:
        experienceLabels[item.experience_level] || item.experience_level,
      salary: item.predicted_salary_usd,
      color: EXP_COLORS[item.experience_level] || "#8b5cf6",
    }));
  }, [selectedRecords]);

  // Chart 3: Salary Range Analysis Data
  const salaryRangeChartData = useMemo(() => {
    return selectedRecords.map((item) => {
      const minVal =
        item.salary_range?.min ??
        Math.max(
          0,
          item.predicted_salary_usd - (item.confidence_margin_mae || 39039)
        );
      const maxVal =
        item.salary_range?.max ??
        item.predicted_salary_usd + (item.confidence_margin_mae || 39039);
      return {
        id: item.id,
        name:
          item.job_title.length > 14
            ? `${item.job_title.slice(0, 12)}...`
            : item.job_title,
        fullName: item.job_title,
        experienceTier:
          experienceLabels[item.experience_level] || item.experience_level,
        min: Math.round(minVal),
        predicted: Math.round(item.predicted_salary_usd),
        max: Math.round(maxVal),
        mae: Math.round(item.confidence_margin_mae || 39039),
      };
    });
  }, [selectedRecords]);

  // Generate multi-profile comparison PDF report
  const handleExportComparisonPdf = useCallback(() => {
    if (!selectedRecords || selectedRecords.length < 2) return;
    const ok = generateComparisonPdf(
      selectedRecords,
      comparisonStats,
      pdfOptions
    );
    if (ok) {
      showNotification(
        "Comparison PDF report generated! Review or save directly in the print dialog."
      );
    }
  }, [selectedRecords, comparisonStats, pdfOptions, showNotification]);

  return (
    <div className="history-wrapper">
      <section className="history-panel">
        {/* Header */}
        <div className="history-header">
          <div className="header-left-col">
            <div className="database-badge">
              <Database size={13} />
              <span>SQLite Persistent Storage</span>
            </div>
            <h2 className="history-title">Prediction History Dashboard</h2>
            <p className="history-subtitle">
              Audit, inspect, and benchmark all historical salary predictions stored
              locally in your ACID-compliant SQLite database.
            </p>
          </div>

          <div className="header-actions-col">
            <button
              className="history-action-btn"
              onClick={handleRefresh}
              disabled={loading}
              title="Refresh database records"
            >
              <RefreshCw size={14} className={loading ? "spin-icon" : ""} />
              <span>{loading ? "Refreshing..." : "Refresh"}</span>
            </button>

            <button
              className="history-action-btn"
              onClick={handleExportCSV}
              disabled={!history.length}
              title="Export all records as CSV"
            >
              <Download size={14} />
              <span>Export CSV</span>
            </button>

            {history.length > 0 && (
              <button
                className="history-action-btn danger"
                onClick={handleClearAll}
                title="Clear all records"
              >
                <Trash2 size={14} />
                <span>Clear History</span>
              </button>
            )}
          </div>
        </div>

        {/* Notifications & Offline Alert */}
        {notification && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "10px 14px",
              background: "rgba(16, 185, 129, 0.12)",
              border: "1px solid rgba(16, 185, 129, 0.3)",
              borderRadius: 10,
              fontSize: 13,
              color: "#34d399",
            }}
          >
            <CheckCircle2 size={15} />
            <span>{notification}</span>
          </div>
        )}

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
            }}
          >
            <AlertCircle size={15} />
            <span>{errorNotice}</span>
          </div>
        )}

        {/* Summary Statistics KPI Cards */}
        <div className="history-stats-grid">
          <div className="history-stat-card">
            <div className="stat-card-top">
              <span>Total Predictions</span>
              <div className="stat-card-icon purple">
                <History size={16} />
              </div>
            </div>
            <h3>{summary ? summary.total_predictions : history.length}</h3>
            <span className="stat-card-sub">
              {isLive ? "Logged in SQLite" : "Local snapshot"}
            </span>
          </div>

          <div className="history-stat-card">
            <div className="stat-card-top">
              <span>Average Predicted</span>
              <div className="stat-card-icon blue">
                <DollarSign size={16} />
              </div>
            </div>
            <h3>
              {summary?.formatted_avg_salary
                ? summary.formatted_avg_salary
                : history.length > 0
                ? `$${Math.round(
                    history.reduce((a, b) => a + b.predicted_salary_usd, 0) /
                      history.length
                  ).toLocaleString()}`
                : "$0"}
            </h3>
            <span className="stat-card-sub">Estimated compensation</span>
          </div>

          <div className="history-stat-card">
            <div className="stat-card-top">
              <span>Salary Range</span>
              <div className="stat-card-icon green">
                <TrendingUp size={16} />
              </div>
            </div>
            <h3>
              {summary && summary.total_predictions > 0
                ? `$${Math.round(summary.min_predicted_salary / 1000)}k - $${Math.round(
                    summary.max_predicted_salary / 1000
                  )}k`
                : history.length > 0
                ? `$${Math.round(
                    Math.min(...history.map((h) => h.predicted_salary_usd)) / 1000
                  )}k - $${Math.round(
                    Math.max(...history.map((h) => h.predicted_salary_usd)) / 1000
                  )}k`
                : "$0 - $0"}
            </h3>
            <span className="stat-card-sub">Min to Max predictions</span>
          </div>

          <div className="history-stat-card">
            <div className="stat-card-top">
              <span>Frequent Role</span>
              <div className="stat-card-icon amber">
                <Sparkles size={16} />
              </div>
            </div>
            <h3
              style={{
                fontSize: 16,
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              {summary?.most_frequent_role ||
                history[0]?.job_title ||
                "None yet"}
            </h3>
            <span className="stat-card-sub">
              Top Level: {summary?.top_experience_level || "SE"}
            </span>
          </div>
        </div>

        {/* Toolbar: Search & Filters */}
        <div className="history-toolbar">
          <div className="search-box">
            <Search size={15} style={{ color: "#64748b" }} />
            <input
              type="text"
              placeholder="Search by job title, location, country..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button
                className="search-clear-btn"
                onClick={() => setSearchQuery("")}
              >
                <X size={14} />
              </button>
            )}
          </div>

          <div className="toolbar-filters">
            <div className="filter-group">
              <span className="filter-label">Level:</span>
              <select
                className="filter-select"
                value={experienceFilter}
                onChange={(e) => setExperienceFilter(e.target.value)}
              >
                <option value="ALL">All Levels</option>
                <option value="EN">Entry (EN)</option>
                <option value="MI">Mid-level (MI)</option>
                <option value="SE">Senior (SE)</option>
                <option value="EX">Executive (EX)</option>
              </select>
            </div>

            <div className="filter-group">
              <span className="filter-label">Remote:</span>
              <select
                className="filter-select"
                value={remoteFilter}
                onChange={(e) => setRemoteFilter(e.target.value)}
              >
                <option value="ALL">All Settings</option>
                <option value="100">100% Remote</option>
                <option value="50">50% Hybrid</option>
                <option value="0">On-site (0%)</option>
              </select>
            </div>

            <div className="filter-group">
              <span className="filter-label">Sort:</span>
              <select
                className="filter-select"
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
              >
                <option value="latest">Latest First</option>
                <option value="salary-desc">Highest Salary</option>
                <option value="salary-asc">Lowest Salary</option>
                <option value="role">Role (A-Z)</option>
              </select>
            </div>

            {filteredHistory.length > 0 && (
              <button
                type="button"
                className={`history-action-btn select-all-btn ${
                  selectedRecordIds.length > 0 ? "active" : ""
                }`}
                onClick={handleSelectAllVisible}
                title="Select up to 4 predictions for comparison"
              >
                <CheckSquare size={13} />
                <span>
                  {selectedRecordIds.length >=
                    Math.min(4, filteredHistory.length) &&
                  filteredHistory.length > 0
                    ? "Deselect All"
                    : "Select (Max 4)"}
                </span>
              </button>
            )}
          </div>
        </div>

        {/* Comparison Selection Bar */}
        {selectedRecordIds.length > 0 && (
          <div className="history-compare-bar">
            <div className="compare-bar-left">
              <div className="compare-bar-badge">
                <Scale size={15} />
                <span>{selectedRecordIds.length} of 4 Selected</span>
              </div>
              <span className="compare-bar-hint">
                {selectedRecordIds.length === 1
                  ? "Select at least 1 more prediction to compare side-by-side."
                  : `Ready to compare ${selectedRecordIds.length} predictions side-by-side.`}
              </span>
            </div>

            <div className="compare-bar-actions">
              <button
                type="button"
                className="compare-clear-btn"
                onClick={handleClearSelection}
                title="Clear current selection"
              >
                <X size={14} />
                <span>Clear</span>
              </button>

              <button
                type="button"
                className={`history-action-btn primary compare-now-btn ${
                  selectedRecordIds.length < 2 ? "disabled" : ""
                }`}
                disabled={selectedRecordIds.length < 2}
                title={
                  selectedRecordIds.length < 2
                    ? "Select at least 2 predictions to compare"
                    : `Compare ${selectedRecordIds.length} predictions side-by-side`
                }
                onClick={() => setShowCompareModal(true)}
              >
                <Scale size={14} />
                <span>Compare ({selectedRecordIds.length})</span>
              </button>
            </div>
          </div>
        )}

        {/* Records List or Empty State */}
        {filteredHistory.length === 0 ? (
          <div className="history-empty-state">
            <div className="empty-icon-wrap">
              <Database size={26} />
            </div>
            <h3>
              {history.length === 0
                ? "No Prediction History Recorded Yet"
                : "No Predictions Match Your Filters"}
            </h3>
            <p>
              {history.length === 0
                ? "Generate salary predictions using the Salary Predictor tool. Every calculation is automatically logged into SQLite with full reproducibility."
                : "Try adjusting your search query, experience tier, or workplace filters to view matching prediction entries."}
            </p>
            {onNavigateToPredictor && (
              <button
                className="history-action-btn primary"
                onClick={onNavigateToPredictor}
                style={{ marginTop: 8 }}
              >
                <Sparkles size={14} />
                <span>Make a Prediction Now</span>
              </button>
            )}
          </div>
        ) : (
          <div className="history-records-container">
            {filteredHistory.map((item) => {
              const expClass = getExperienceClass(item.experience_level);
              const isDeleting = deletingId === item.id;
              const isSelected = selectedRecordIds.includes(item.id);

              return (
                <div
                  key={item.id}
                  className={`history-record-card ${isSelected ? "selected" : ""}`}
                  style={{ opacity: isDeleting ? 0.4 : 1 }}
                >
                  <div
                    className="history-checkbox-wrap"
                    onClick={(e) => toggleSelectRecord(item.id, e)}
                    title={
                      isSelected
                        ? "Deselect prediction"
                        : selectedRecordIds.length >= 4
                        ? "Maximum 4 predictions selected"
                        : "Select for comparison"
                    }
                  >
                    <button
                      type="button"
                      className={`history-checkbox ${isSelected ? "checked" : ""}`}
                      aria-label={`Select prediction #${item.id} for comparison`}
                    >
                      {isSelected ? <CheckSquare size={18} /> : <Square size={18} />}
                    </button>
                  </div>

                  <div className="record-main-col">
                    <div className="record-top-row">
                      <span className="record-role-title">{item.job_title}</span>
                      <span className={`badge-pill ${expClass}`}>
                        {experienceLabels[item.experience_level] || item.experience_level}
                      </span>
                      <span className="badge-pill neutral">
                        {item.employment_type || "FT"}
                      </span>
                      <span className="badge-pill neutral">
                        {item.remote_ratio === 100
                          ? "Remote"
                          : item.remote_ratio === 50
                          ? "Hybrid"
                          : "On-site"}
                      </span>
                    </div>

                    <div className="record-meta-row">
                      <span className="meta-item">
                        <MapPin size={12} />
                        Loc: {item.company_location || "US"} (Res:{" "}
                        {item.employee_residence || "US"})
                      </span>
                      <span className="meta-item">
                        <Building2 size={12} />
                        Scale: {item.company_size || "M"}
                      </span>
                      <span className="meta-item">
                        <Laptop size={12} />
                        Year: {item.work_year || 2023}
                      </span>
                      <span className="meta-item">
                        <Calendar size={12} />
                        {formatDate(item.timestamp)}
                      </span>
                    </div>
                  </div>

                  <div className="record-salary-col">
                    <span className="record-salary-amount">
                      {item.formatted_salary ||
                        `$${Math.round(item.predicted_salary_usd).toLocaleString()}`}
                    </span>
                    <span className="record-salary-range">
                      {item.salary_range
                        ? `${item.salary_range.formatted_min} - ${item.salary_range.formatted_max}`
                        : "±$39,000 Range"}
                    </span>
                  </div>

                  <div className="history-row-actions">
                    <button
                      className="history-view"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleViewDetails(item.id);
                      }}
                      title="View details"
                    >
                      <Eye size={16} />
                    </button>

                    <button
                      className="history-delete"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(item.id);
                      }}
                      title="Delete prediction"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Footer info */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            paddingTop: 12,
            borderTop: "1px solid #1e293b",
            fontSize: 12,
            color: "#64748b",
            flexWrap: "wrap",
            gap: 10,
          }}
        >
          <span>
            Showing {filteredHistory.length} of {history.length} stored predictions
          </span>
          <span>Database: backend/data/prediction_history.db</span>
        </div>

        {/* Step 3: Details modal */}
        {(detailsLoading || detailsError || selectedPrediction) && (
          <div className="prediction-modal-overlay">
            <div className={`prediction-modal ${detailTab === "xai" ? "modal-expanded" : ""}`}>
              <div className="prediction-modal-header">
                <div>
                  <span className="history-eyebrow">PREDICTION DETAILS</span>
                  <h2>{selectedPrediction?.job_title || "Loading..."}</h2>
                </div>

                <button
                  className="prediction-modal-close"
                  onClick={() => {
                    setSelectedPrediction(null);
                    setDetailsError("");
                    setDetailTab("overview");
                  }}
                >
                  <X size={20} />
                </button>
              </div>

              {detailsLoading && <p>Loading prediction details...</p>}

              {detailsError && (
                <div className="history-error">{detailsError}</div>
              )}

              {selectedPrediction && (
                <>
                  <div className="prediction-modal-tabs">
                    <button
                      type="button"
                      className={`prediction-tab-btn ${detailTab === "overview" ? "active" : ""}`}
                      onClick={() => setDetailTab("overview")}
                    >
                      <FileText size={15} />
                      <span>Profile & Range</span>
                    </button>

                    <button
                      type="button"
                      className={`prediction-tab-btn ${detailTab === "xai" ? "active" : ""}`}
                      onClick={() => setDetailTab("xai")}
                    >
                      <Sparkles size={15} />
                      <span>Explain Prediction (SHAP XAI) 💡</span>
                    </button>
                  </div>

                  {detailTab === "overview" ? (
                    <>
                      <div className="prediction-detail-salary">
                        <span>ESTIMATED ANNUAL SALARY</span>
                        <strong>
                          {selectedPrediction.formatted_salary ||
                            formatMoney(selectedPrediction.predicted_salary_usd)}
                        </strong>
                        <p>USD / year</p>
                      </div>

                      <div className="prediction-detail-grid">
                        <div>
                          <span>Lower estimate</span>
                          <strong>{formatMoney(selectedPrediction.salary_range?.min)}</strong>
                        </div>

                        <div>
                          <span>Upper estimate</span>
                          <strong>{formatMoney(selectedPrediction.salary_range?.max)}</strong>
                        </div>

                        <div>
                          <span>MAE margin</span>
                          <strong>{formatMoney(selectedPrediction.confidence_margin_mae)}</strong>
                        </div>

                        <div>
                          <span>Experience</span>
                          <strong>
                            {experienceLabels[selectedPrediction.experience_level] ||
                              selectedPrediction.experience_level}
                          </strong>
                        </div>

                        <div>
                          <span>Company location</span>
                          <strong>{selectedPrediction.company_location}</strong>
                        </div>

                        <div>
                          <span>Employee residence</span>
                          <strong>{selectedPrediction.employee_residence}</strong>
                        </div>

                        <div>
                          <span>Employment type</span>
                          <strong>{selectedPrediction.employment_type}</strong>
                        </div>

                        <div>
                          <span>Company size</span>
                          <strong>{selectedPrediction.company_size}</strong>
                        </div>

                        <div>
                          <span>Remote work</span>
                          <strong>{selectedPrediction.remote_ratio}%</strong>
                        </div>

                        <div>
                          <span>Work year</span>
                          <strong>{selectedPrediction.work_year}</strong>
                        </div>
                      </div>

                      <div className="prediction-model-version">
                        <span>Model version</span>
                        <strong>{selectedPrediction.model_version}</strong>
                      </div>

                      <div
                        className="xai-callout-banner"
                        onClick={() => setDetailTab("xai")}
                        title="Click to view full SHAP breakdown"
                      >
                        <div className="xai-callout-left">
                          <Sparkles size={18} className="xai-callout-icon" />
                          <div>
                            <strong>Why was this salary predicted?</strong>
                            <p>
                              Explore exact TreeSHAP feature attributions, waterfall steps, and key drivers.
                            </p>
                          </div>
                        </div>
                        <span className="xai-callout-link">Open SHAP &rarr;</span>
                      </div>
                    </>
                  ) : (
                    <div className="modal-xai-container">
                      <ShapExplanationView shapData={selectedPrediction.shap_explanation} />
                    </div>
                  )}

                  <div className="prediction-modal-date">
                    Generated:{" "}
                    {selectedPrediction.timestamp
                      ? new Date(selectedPrediction.timestamp).toLocaleString()
                      : "—"}
                  </div>

                  <div className="prediction-modal-actions">
                    <button
                      type="button"
                      className="history-action-btn primary"
                      onClick={() => setShowPdfCustomizer(true)}
                      title="Tailor and export printable PDF report"
                    >
                      <FileText size={15} />
                      <span>Customize & Download PDF 📄</span>
                    </button>

                    <button
                      type="button"
                      className="history-action-btn"
                      onClick={() => generatePdfReport(selectedPrediction, pdfOptions)}
                      title="Quick print without customizing"
                    >
                      <Printer size={15} />
                      <span>Quick Print</span>
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* PDF Customization Modal */}
        {showPdfCustomizer && selectedPrediction && (
          <div
            className="pdf-customizer-overlay"
            onClick={() => setShowPdfCustomizer(false)}
          >
            <div
              className="pdf-customizer-modal"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="customizer-header">
                <div>
                  <span className="history-eyebrow">PDF REPORT CUSTOMIZER 📄</span>
                  <h3>Tailor Executive Compensation Report</h3>
                </div>
                <button
                  className="prediction-modal-close"
                  onClick={() => setShowPdfCustomizer(false)}
                >
                  <X size={18} />
                </button>
              </div>

              <div className="customizer-body">
                <div className="customizer-row">
                  <div className="customizer-field">
                    <label>Candidate / Recipient Name</label>
                    <input
                      type="text"
                      value={pdfOptions.candidateName}
                      onChange={(e) =>
                        setPdfOptions((prev) => ({
                          ...prev,
                          candidateName: e.target.value,
                        }))
                      }
                      placeholder="e.g. Alex Morgan"
                    />
                  </div>

                  <div className="customizer-field">
                    <label>Target Organization / Dept</label>
                    <input
                      type="text"
                      value={pdfOptions.targetCompany}
                      onChange={(e) =>
                        setPdfOptions((prev) => ({
                          ...prev,
                          targetCompany: e.target.value,
                        }))
                      }
                      placeholder="e.g. Acme AI Labs"
                    />
                  </div>
                </div>

                <div className="customizer-row">
                  <div className="customizer-field">
                    <label>Report Header Title</label>
                    <input
                      type="text"
                      value={pdfOptions.reportTitle}
                      onChange={(e) =>
                        setPdfOptions((prev) => ({
                          ...prev,
                          reportTitle: e.target.value,
                        }))
                      }
                      placeholder="e.g. Executive Compensation Review"
                    />
                  </div>

                  <div className="customizer-field">
                    <label>Accent Color Palette</label>
                    <select
                      value={pdfOptions.themeColor}
                      onChange={(e) =>
                        setPdfOptions((prev) => ({
                          ...prev,
                          themeColor: e.target.value,
                        }))
                      }
                    >
                      <option value="purple">Modern Purple (SalaryIQ Default)</option>
                      <option value="blue">Executive Sapphire Blue</option>
                      <option value="emerald">High-Growth Emerald Green</option>
                    </select>
                  </div>
                </div>

                <div className="customizer-field full">
                  <label>Custom Strategic Notes (Appears on Report)</label>
                  <textarea
                    rows={3}
                    value={pdfOptions.customNotes}
                    onChange={(e) =>
                      setPdfOptions((prev) => ({
                        ...prev,
                        customNotes: e.target.value,
                      }))
                    }
                    placeholder="Enter confidential notes, counter-offer tactics, or justification highlights..."
                  />
                </div>

                <div className="customizer-toggles-section">
                  <span className="toggle-heading">Included Report Sections</span>
                  <div className="toggles-grid">
                    <label className="checkbox-item">
                      <input
                        type="checkbox"
                        checked={pdfOptions.includeConfidence}
                        onChange={(e) =>
                          setPdfOptions((prev) => ({
                            ...prev,
                            includeConfidence: e.target.checked,
                          }))
                        }
                      />
                      <span>MAE Confidence Range (±$39,039)</span>
                    </label>

                    <label className="checkbox-item">
                      <input
                        type="checkbox"
                        checked={pdfOptions.includeNegotiation}
                        onChange={(e) =>
                          setPdfOptions((prev) => ({
                            ...prev,
                            includeNegotiation: e.target.checked,
                          }))
                        }
                      />
                      <span>Salary Negotiation Playbook & Advice</span>
                    </label>

                    <label className="checkbox-item">
                      <input
                        type="checkbox"
                        checked={pdfOptions.includeExplainability}
                        onChange={(e) =>
                          setPdfOptions((prev) => ({
                            ...prev,
                            includeExplainability: e.target.checked,
                          }))
                        }
                      />
                      <span>Random Forest Model Audit & Feature Space</span>
                    </label>
                  </div>
                </div>
              </div>

              <div className="customizer-footer">
                <button
                  type="button"
                  className="history-action-btn primary"
                  onClick={handleGeneratePdf}
                >
                  <Printer size={15} />
                  <span>Generate & Save PDF</span>
                </button>
                <button
                  type="button"
                  className="history-action-btn"
                  onClick={() => setShowPdfCustomizer(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Step 2: Comparison Dashboard Modal */}
        {showCompareModal && selectedRecords.length >= 2 && comparisonStats && (
          <div
            className="compare-modal-overlay"
            onClick={() => setShowCompareModal(false)}
          >
            <div
              className="compare-modal-window"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div className="compare-modal-header">
                <div className="compare-modal-title-col">
                  <div className="compare-eyebrow">
                    <Columns3 size={14} />
                    <span>PREDICTION COMPARISON DASHBOARD</span>
                  </div>
                  <h2>Side-by-Side Compensation Benchmark</h2>
                  <p>
                    Directly comparing {selectedRecords.length} historical machine
                    learning salary estimates.
                  </p>
                </div>

                <div className="compare-header-actions">
                  <div className="compare-view-switcher">
                    <button
                      type="button"
                      className={`view-switcher-btn ${
                        compareViewMode === "all" ? "active" : ""
                      }`}
                      onClick={() => setCompareViewMode("all")}
                      title="Show both profile cards and visual charts"
                    >
                      <Layers size={13} />
                      <span>All Views</span>
                    </button>
                    <button
                      type="button"
                      className={`view-switcher-btn ${
                        compareViewMode === "grid" ? "active" : ""
                      }`}
                      onClick={() => setCompareViewMode("grid")}
                      title="Show side-by-side profile cards"
                    >
                      <Columns3 size={13} />
                      <span>Profiles</span>
                    </button>
                    <button
                      type="button"
                      className={`view-switcher-btn ${
                        compareViewMode === "charts" ? "active" : ""
                      }`}
                      onClick={() => setCompareViewMode("charts")}
                      title="Show 3 visual analytics charts"
                    >
                      <BarChart3 size={13} />
                      <span>Charts (3)</span>
                    </button>
                  </div>

                  <button
                    type="button"
                    className="history-action-btn primary compare-export-pdf-btn"
                    onClick={handleExportComparisonPdf}
                    title="Export full comparison report to printable PDF"
                  >
                    <FileText size={14} />
                    <span>Export Comparison PDF 📄</span>
                  </button>

                  <button
                    className="prediction-modal-close"
                    onClick={() => setShowCompareModal(false)}
                    title="Close comparison"
                  >
                    <X size={20} />
                  </button>
                </div>
              </div>

              {/* Summary Stats / Differences Ribbon */}
              <div className="compare-stats-ribbon">
                <div className="compare-stat-pill">
                  <span className="pill-label">Compared Profiles</span>
                  <strong className="pill-value">
                    {selectedRecords.length} Roles
                  </strong>
                </div>

                <div className="compare-stat-pill">
                  <span className="pill-label">Highest Predicted</span>
                  <strong className="pill-value highlight-green">
                    {formatMoney(comparisonStats.maxSalary)}
                  </strong>
                  <span className="pill-sub">
                    {comparisonStats.highestRecord?.job_title}
                  </span>
                </div>

                <div className="compare-stat-pill">
                  <span className="pill-label">Lowest Predicted</span>
                  <strong className="pill-value highlight-amber">
                    {formatMoney(comparisonStats.minSalary)}
                  </strong>
                  <span className="pill-sub">
                    {comparisonStats.lowestRecord?.job_title}
                  </span>
                </div>

                <div className="compare-stat-pill">
                  <span className="pill-label">Salary Difference</span>
                  <strong className="pill-value highlight-purple">
                    +{formatMoney(comparisonStats.diff)}
                  </strong>
                  <span className="pill-sub">
                    +{comparisonStats.pctDiff}% spread
                  </span>
                </div>

                <div className="compare-stat-pill">
                  <span className="pill-label">Group Average</span>
                  <strong className="pill-value">
                    {formatMoney(comparisonStats.avgSalary)}
                  </strong>
                  <span className="pill-sub">USD / year</span>
                </div>
              </div>

              {/* Side-by-Side Comparison Columns Container */}
              {(compareViewMode === "all" || compareViewMode === "grid") && (
                <div
                  className="compare-grid-container"
                  style={{
                    gridTemplateColumns: `repeat(${selectedRecords.length}, minmax(260px, 1fr))`,
                  }}
                >
                {selectedRecords.map((item) => {
                  const isHighest = item.id === comparisonStats.highestRecord?.id;
                  const isLowest = item.id === comparisonStats.lowestRecord?.id;
                  const salaryDelta =
                    item.predicted_salary_usd - comparisonStats.minSalary;
                  const deltaPct =
                    comparisonStats.minSalary > 0
                      ? Math.round(
                          (salaryDelta / comparisonStats.minSalary) * 100
                        )
                      : 0;
                  const expClass = getExperienceClass(item.experience_level);

                  return (
                    <div
                      key={item.id}
                      className={`compare-card-column ${
                        isHighest
                          ? "highest-card"
                          : isLowest
                          ? "lowest-card"
                          : ""
                      }`}
                    >
                      {/* Column Header */}
                      <div className="compare-card-header">
                        <div className="compare-card-top-info">
                          <span className={`badge-pill ${expClass}`}>
                            {experienceLabels[item.experience_level] ||
                              item.experience_level}
                          </span>
                          <span className="compare-record-id">#{item.id}</span>
                        </div>
                        <button
                          type="button"
                          className="compare-remove-card-btn"
                          onClick={() => handleRemoveFromCompare(item.id)}
                          title="Remove from comparison"
                        >
                          <X size={14} />
                        </button>
                      </div>

                      <h3 className="compare-job-title">{item.job_title}</h3>

                      {/* Salary Display Box */}
                      <div className="compare-salary-box">
                        <div className="compare-salary-badge-row">
                          {isHighest && (
                            <span className="compare-badge peak">
                              ★ Highest Estimate
                            </span>
                          )}
                          {isLowest && selectedRecords.length > 1 && (
                            <span className="compare-badge baseline">
                              Base Benchmark
                            </span>
                          )}
                          {!isHighest && !isLowest && (
                            <span className="compare-badge mid">Mid-Tier</span>
                          )}
                        </div>

                        <div className="compare-salary-main">
                          <strong>
                            {item.formatted_salary ||
                              formatMoney(item.predicted_salary_usd)}
                          </strong>
                          <span>USD / year</span>
                        </div>

                        <div className="compare-delta-row">
                          {salaryDelta > 0 ? (
                            <span className="delta-positive">
                              <ArrowUpRight size={13} />
                              +{formatMoney(salaryDelta)} (+{deltaPct}%) vs base
                            </span>
                          ) : (
                            <span className="delta-neutral">
                              Baseline Benchmark
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Estimated Range & MAE */}
                      <div className="compare-range-box">
                        <div className="range-label-row">
                          <span>Confidence Range (±MAE)</span>
                          <strong>
                            ±{formatMoney(item.confidence_margin_mae)}
                          </strong>
                        </div>
                        <div className="range-minmax-row">
                          <span>{formatMoney(item.salary_range?.min)}</span>
                          <div className="range-track-mini">
                            <div className="range-fill-mini" />
                          </div>
                          <span>{formatMoney(item.salary_range?.max)}</span>
                        </div>
                      </div>

                      {/* Detailed Attribute Rows */}
                      <div className="compare-attributes-list">
                        <div className="compare-attr-row">
                          <span className="attr-title">Experience</span>
                          <span className="attr-val">
                            {experienceLabels[item.experience_level] ||
                              item.experience_level}
                          </span>
                        </div>

                        <div className="compare-attr-row">
                          <span className="attr-title">Company Location</span>
                          <span className="attr-val">
                            {item.company_location}
                          </span>
                        </div>

                        <div className="compare-attr-row">
                          <span className="attr-title">
                            Employee Residence
                          </span>
                          <span className="attr-val">
                            {item.employee_residence}
                          </span>
                        </div>

                        <div className="compare-attr-row">
                          <span className="attr-title">Work Setting</span>
                          <span className="attr-val">
                            {item.remote_ratio === 100
                              ? "100% Remote"
                              : item.remote_ratio === 50
                              ? "50% Hybrid"
                              : "On-site (0%)"}
                          </span>
                        </div>

                        <div className="compare-attr-row">
                          <span className="attr-title">Employment Type</span>
                          <span className="attr-val">
                            {item.employment_type}
                          </span>
                        </div>

                        <div className="compare-attr-row">
                          <span className="attr-title">Company Size</span>
                          <span className="attr-val">
                            {item.company_size === "L"
                              ? "Large (250+)"
                              : item.company_size === "M"
                              ? "Medium (50-249)"
                              : "Small (<50)"}
                          </span>
                        </div>

                        <div className="compare-attr-row">
                          <span className="attr-title">Work Year</span>
                          <span className="attr-val">{item.work_year}</span>
                        </div>

                        <div className="compare-attr-row">
                          <span className="attr-title">Model Version</span>
                          <span className="attr-val font-mono">
                            {item.model_version}
                          </span>
                        </div>

                        <div className="compare-attr-row">
                          <span className="attr-title">Predicted Date</span>
                          <span className="attr-val">
                            {formatDate(item.timestamp)}
                          </span>
                        </div>
                      </div>

                      {/* Card Action */}
                      <div className="compare-card-footer">
                        <button
                          type="button"
                          className="compare-card-action-btn"
                          onClick={() => {
                            setShowCompareModal(false);
                            handleViewDetails(item.id);
                          }}
                        >
                          <Eye size={13} />
                          <span>View Full Audit</span>
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
              )}

              {/* Step 3: Visual Analytics Charts Section */}
              {(compareViewMode === "all" || compareViewMode === "charts") && (
                <div className="compare-charts-section">
                  <div className="compare-charts-header">
                    <div className="charts-header-left">
                      <div className="charts-eyebrow">
                        <BarChart3 size={13} />
                        <span>RECHARTS VISUAL ANALYTICS</span>
                      </div>
                      <h3>Comparative Distribution & Salary Analytics</h3>
                      <p>
                        Visual benchmark across {selectedRecords.length} selected job profiles. Updates automatically in real time.
                      </p>
                    </div>
                    <div className="charts-header-badge">
                      <span className="live-dot" />
                      Live Model Insights
                    </div>
                  </div>

                  <div className="compare-charts-grid">
                    {/* Chart 1: Salary Comparison */}
                    <div className="compare-chart-card">
                      <div className="chart-card-top">
                        <div className="chart-card-title-wrap">
                          <span className="chart-num-badge">Chart 1</span>
                          <h4>Predicted Salary Comparison</h4>
                        </div>
                        <span className="chart-card-sub">
                          Direct comparison of predicted annual compensation
                        </span>
                      </div>

                      <div className="chart-card-body">
                        <ResponsiveContainer width="100%" height={260}>
                          <BarChart
                            data={salaryComparisonChartData}
                            margin={{ top: 15, right: 15, left: 10, bottom: 25 }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke="#1e293b"
                              vertical={false}
                            />
                            <XAxis
                              dataKey="name"
                              stroke="#64748b"
                              tick={{ fill: "#94a3b8", fontSize: 11 }}
                            />
                            <YAxis
                              tickFormatter={(v) => `$${Math.round(v / 1000)}k`}
                              stroke="#64748b"
                              tick={{ fill: "#94a3b8", fontSize: 11 }}
                            />
                            <Tooltip content={<CompareChartTooltip />} />
                            <Bar
                              dataKey="salary"
                              name="Predicted Salary"
                              radius={[6, 6, 0, 0]}
                            >
                              {salaryComparisonChartData.map((entry) => (
                                <Cell key={entry.id} fill={entry.fill} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="chart-legend-custom">
                        <span className="legend-item">
                          <span className="legend-dot green" /> Highest Estimate
                        </span>
                        <span className="legend-item">
                          <span className="legend-dot purple" /> Mid-Tier
                        </span>
                        <span className="legend-item">
                          <span className="legend-dot amber" /> Base Benchmark
                        </span>
                      </div>
                    </div>

                    {/* Chart 2: Experience-Level Comparison */}
                    <div className="compare-chart-card">
                      <div className="chart-card-top">
                        <div className="chart-card-title-wrap">
                          <span className="chart-num-badge">Chart 2</span>
                          <h4>Experience-Level Comparison</h4>
                        </div>
                        <span className="chart-card-sub">
                          Salaries ordered by seniority (EN → MI → SE → EX)
                        </span>
                      </div>

                      <div className="chart-card-body">
                        <ResponsiveContainer width="100%" height={260}>
                          <BarChart
                            data={experienceComparisonChartData}
                            margin={{ top: 15, right: 15, left: 10, bottom: 25 }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke="#1e293b"
                              vertical={false}
                            />
                            <XAxis
                              dataKey="label"
                              stroke="#64748b"
                              tick={{ fill: "#94a3b8", fontSize: 11 }}
                            />
                            <YAxis
                              tickFormatter={(v) => `$${Math.round(v / 1000)}k`}
                              stroke="#64748b"
                              tick={{ fill: "#94a3b8", fontSize: 11 }}
                            />
                            <Tooltip content={<CompareChartTooltip />} />
                            <Bar
                              dataKey="salary"
                              name="Predicted Salary"
                              radius={[6, 6, 0, 0]}
                            >
                              {experienceComparisonChartData.map((entry) => (
                                <Cell key={entry.id} fill={entry.color} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="chart-legend-custom">
                        <span className="legend-item">
                          <span className="legend-dot exp-en" /> Entry (EN)
                        </span>
                        <span className="legend-item">
                          <span className="legend-dot exp-mi" /> Mid (MI)
                        </span>
                        <span className="legend-item">
                          <span className="legend-dot exp-se" /> Senior (SE)
                        </span>
                        <span className="legend-item">
                          <span className="legend-dot exp-ex" /> Exec (EX)
                        </span>
                      </div>
                    </div>

                    {/* Chart 3: Salary Range Analysis */}
                    <div className="compare-chart-card full-width">
                      <div className="chart-card-top">
                        <div className="chart-card-title-wrap">
                          <span className="chart-num-badge">Chart 3</span>
                          <h4>Salary Range Analysis (Min vs Predicted vs Max)</h4>
                        </div>
                        <span className="chart-card-sub">
                          Lower estimate, model prediction, and upper estimate with ±MAE confidence band
                        </span>
                      </div>

                      <div className="chart-card-body">
                        <ResponsiveContainer width="100%" height={280}>
                          <BarChart
                            data={salaryRangeChartData}
                            margin={{ top: 15, right: 20, left: 10, bottom: 25 }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke="#1e293b"
                              vertical={false}
                            />
                            <XAxis
                              dataKey="name"
                              stroke="#64748b"
                              tick={{ fill: "#94a3b8", fontSize: 11 }}
                            />
                            <YAxis
                              tickFormatter={(v) => `$${Math.round(v / 1000)}k`}
                              stroke="#64748b"
                              tick={{ fill: "#94a3b8", fontSize: 11 }}
                            />
                            <Tooltip content={<CompareChartTooltip />} />
                            <Legend
                              wrapperStyle={{
                                paddingTop: 10,
                                fontSize: 12,
                                color: "#94a3b8",
                              }}
                            />
                            <Bar
                              dataKey="min"
                              name="Lower Range (Min)"
                              fill="#475569"
                              radius={[5, 5, 0, 0]}
                            />
                            <Bar
                              dataKey="predicted"
                              name="Predicted Salary"
                              fill="#8b5cf6"
                              radius={[5, 5, 0, 0]}
                            />
                            <Bar
                              dataKey="max"
                              name="Upper Range (Max)"
                              fill="#38bdf8"
                              radius={[5, 5, 0, 0]}
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Modal Footer */}
              <div className="compare-modal-footer">
                <div className="compare-footer-left">
                  <span>
                    📄 Professional A4 print-ready comparison report with vector charts & MAE empirical bounds.
                  </span>
                </div>

                <div className="compare-footer-right">
                  <button
                    type="button"
                    className="history-action-btn primary"
                    onClick={handleExportComparisonPdf}
                    title="Export full comparison report to printable PDF"
                  >
                    <FileText size={14} />
                    <span>Export Comparison PDF 📄</span>
                  </button>

                  <button
                    type="button"
                    className="history-action-btn"
                    onClick={() => setShowCompareModal(false)}
                  >
                    Close Comparison
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
