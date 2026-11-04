import React from "react";
import "./Skeletons.css";

export function AnalyticsSkeleton() {
  return (
    <div className="sk-container">
      {/* Header */}
      <div className="sk-header">
        <div className="sk-shimmer" style={{ width: "240px", height: "30px", marginBottom: "10px" }} />
        <div className="sk-shimmer" style={{ width: "420px", height: "16px" }} />
      </div>

      {/* 4 Top KPI Metric Cards */}
      <div className="sk-grid-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="sk-card">
            <div className="sk-shimmer" style={{ width: "110px", height: "14px", marginBottom: "12px" }} />
            <div className="sk-shimmer" style={{ width: "140px", height: "32px", marginBottom: "8px" }} />
            <div className="sk-shimmer" style={{ width: "90px", height: "12px" }} />
          </div>
        ))}
      </div>

      {/* Charts Grid */}
      <div className="sk-chart-grid">
        <div className="sk-chart-panel">
          <div className="sk-shimmer" style={{ width: "180px", height: "20px", marginBottom: "18px" }} />
          <div className="sk-shimmer" style={{ width: "100%", height: "230px" }} />
        </div>
        <div className="sk-chart-panel">
          <div className="sk-shimmer" style={{ width: "200px", height: "20px", marginBottom: "18px" }} />
          <div className="sk-shimmer" style={{ width: "100%", height: "230px" }} />
        </div>
      </div>
    </div>
  );
}

export function FeatureImportanceSkeleton() {
  return (
    <div className="sk-container">
      <div className="sk-header">
        <div className="sk-shimmer" style={{ width: "260px", height: "30px", marginBottom: "10px" }} />
        <div className="sk-shimmer" style={{ width: "460px", height: "16px" }} />
      </div>

      <div className="sk-card" style={{ padding: "24px" }}>
        <div className="sk-shimmer" style={{ width: "180px", height: "18px", marginBottom: "20px" }} />
        {[92, 85, 78, 65, 52, 44, 38, 25].map((width, idx) => (
          <div key={idx} style={{ marginBottom: "16px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
              <div className="sk-shimmer" style={{ width: "140px", height: "14px" }} />
              <div className="sk-shimmer" style={{ width: "50px", height: "14px" }} />
            </div>
            <div className="sk-shimmer" style={{ width: `${width}%`, height: "12px", borderRadius: "999px" }} />
          </div>
        ))}
      </div>
    </div>
  );
}

export function PredictionHistorySkeleton() {
  return (
    <div className="sk-container">
      <div className="sk-header">
        <div className="sk-shimmer" style={{ width: "240px", height: "30px", marginBottom: "10px" }} />
        <div className="sk-shimmer" style={{ width: "380px", height: "16px" }} />
      </div>

      {/* Filter / Search Bar Skeleton */}
      <div className="sk-card" style={{ display: "flex", gap: "12px", marginBottom: "20px", padding: "14px 18px" }}>
        <div className="sk-shimmer" style={{ flex: 1, height: "36px" }} />
        <div className="sk-shimmer" style={{ width: "130px", height: "36px" }} />
        <div className="sk-shimmer" style={{ width: "130px", height: "36px" }} />
      </div>

      {/* Table Skeleton */}
      <div className="sk-table-container">
        <div className="sk-table-row" style={{ borderBottom: "1px solid rgba(255, 255, 255, 0.08)" }}>
          <div className="sk-shimmer" style={{ width: "80px", height: "16px" }} />
          <div className="sk-shimmer" style={{ width: "160px", height: "16px" }} />
          <div className="sk-shimmer" style={{ width: "120px", height: "16px" }} />
          <div className="sk-shimmer" style={{ width: "140px", height: "16px" }} />
          <div className="sk-shimmer" style={{ flex: 1, height: "16px" }} />
        </div>
        {[1, 2, 3, 4, 5, 6].map((i) => (
          <div key={i} className="sk-table-row">
            <div className="sk-shimmer" style={{ width: "60px", height: "20px", borderRadius: "4px" }} />
            <div className="sk-shimmer" style={{ width: "170px", height: "16px" }} />
            <div className="sk-shimmer" style={{ width: "100px", height: "16px" }} />
            <div className="sk-shimmer" style={{ width: "130px", height: "20px", borderRadius: "4px" }} />
            <div className="sk-shimmer" style={{ width: "110px", height: "14px" }} />
            <div className="sk-shimmer" style={{ width: "80px", height: "30px", borderRadius: "6px", marginLeft: "auto" }} />
          </div>
        ))}
      </div>
    </div>
  );
}

export function ShapDashboardSkeleton() {
  return (
    <div className="sk-container">
      <div className="sk-header">
        <div className="sk-shimmer" style={{ width: "300px", height: "30px", marginBottom: "10px" }} />
        <div className="sk-shimmer" style={{ width: "480px", height: "16px" }} />
      </div>

      {/* 3 Metric Cards */}
      <div className="sk-grid-4" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))" }}>
        {[1, 2, 3].map((i) => (
          <div key={i} className="sk-card">
            <div className="sk-shimmer" style={{ width: "130px", height: "14px", marginBottom: "12px" }} />
            <div className="sk-shimmer" style={{ width: "160px", height: "34px", marginBottom: "8px" }} />
            <div className="sk-shimmer" style={{ width: "180px", height: "12px" }} />
          </div>
        ))}
      </div>

      {/* Main Waterfall Chart Card */}
      <div className="sk-chart-panel" style={{ minHeight: "380px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "20px" }}>
          <div className="sk-shimmer" style={{ width: "240px", height: "22px" }} />
          <div className="sk-shimmer" style={{ width: "140px", height: "30px", borderRadius: "6px" }} />
        </div>
        <div className="sk-shimmer" style={{ width: "100%", height: "290px" }} />
      </div>
    </div>
  );
}

export function ShapExplanationSkeleton({ compact = false }) {
  return (
    <div
      style={{
        padding: compact ? "16px" : "22px",
        background: "rgba(15, 23, 42, 0.75)",
        border: "1px solid rgba(255, 255, 255, 0.08)",
        borderRadius: "12px",
        backdropFilter: "blur(8px)",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "14px" }}>
        <div className="sk-shimmer" style={{ width: "180px", height: "18px" }} />
        <div className="sk-shimmer" style={{ width: "80px", height: "18px", borderRadius: "4px" }} />
      </div>
      <div className="sk-shimmer" style={{ width: "100%", height: compact ? "160px" : "240px", marginBottom: "12px" }} />
      <div style={{ display: "flex", gap: "8px" }}>
        <div className="sk-shimmer" style={{ flex: 1, height: "14px" }} />
        <div className="sk-shimmer" style={{ flex: 1, height: "14px" }} />
      </div>
    </div>
  );
}

