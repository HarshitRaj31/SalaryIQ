import React, { Component } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error(
      `[ErrorBoundary] Caught error in ${this.props.componentName || "lazy component"}:`,
      error,
      errorInfo
    );
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div
          style={{
            padding: "36px 24px",
            margin: "20px 0",
            background: "rgba(15, 23, 42, 0.85)",
            border: "1px solid rgba(239, 68, 68, 0.35)",
            borderRadius: "14px",
            textAlign: "center",
            backdropFilter: "blur(10px)",
          }}
        >
          <div
            style={{
              width: "48px",
              height: "48px",
              margin: "0 auto 16px",
              borderRadius: "50%",
              background: "rgba(239, 68, 68, 0.15)",
              color: "#f87171",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <AlertTriangle size={24} />
          </div>
          <h3
            style={{
              margin: "0 0 8px",
              fontSize: "17px",
              color: "#f8fafc",
              fontWeight: 700,
            }}
          >
            Unable to load {this.props.componentName || "Module"}
          </h3>
          <p
            style={{
              margin: "0 0 20px",
              fontSize: "13.5px",
              color: "#94a3b8",
              maxWidth: "460px",
              marginLeft: "auto",
              marginRight: "auto",
              lineHeight: 1.5,
            }}
          >
            A network or resource error occurred while loading this section.
            The rest of your SalaryIQ workspace remains unaffected.
          </p>
          <button
            type="button"
            onClick={this.handleRetry}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              background: "linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)",
              border: "1px solid #3b82f6",
              color: "#ffffff",
              padding: "9px 20px",
              borderRadius: "8px",
              fontSize: "13px",
              fontWeight: 600,
              cursor: "pointer",
              transition: "all 0.15s ease",
            }}
          >
            <RefreshCw size={15} />
            Retry Loading
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

