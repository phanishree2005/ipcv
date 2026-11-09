"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/", label: "Dashboard", icon: "⬡" },
  { href: "/search", label: "Forensic Search", icon: "🔍" },
  { href: "/inference", label: "Live & Video Re-ID", icon: "📹" },
  { href: "/datasets", label: "Datasets", icon: "🗄" },
  { href: "/topology", label: "Camera Topology", icon: "🌐" },
  { href: "/models", label: "Models", icon: "🧠" },
  { href: "/experiments", label: "Experiments", icon: "⚗" },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">M²</div>
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)" }}>M²-ReID</div>
          <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 1 }}>Research Platform</div>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section-title">Navigation</div>
        {NAV.map(({ href, label, icon }) => (
          <Link
            key={href}
            href={href}
            className={`nav-item ${pathname === href ? "active" : ""}`}
          >
            <span style={{ fontSize: 14, width: 18, textAlign: "center" }}>{icon}</span>
            {label}
          </Link>
        ))}

        <div className="nav-section-title" style={{ marginTop: 24 }}>System</div>
        <Link href="/settings" className={`nav-item ${pathname === "/settings" ? "active" : ""}`}>
          <span style={{ fontSize: 14, width: 18, textAlign: "center" }}>⚙</span>
          Settings
        </Link>
      </nav>

      <div style={{
        padding: "16px 20px",
        borderTop: "1px solid var(--border)",
        display: "flex", alignItems: "center", gap: 8,
      }}>
        <div className="pulse-dot" />
        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>Backend connected</span>
      </div>
    </aside>
  );
}
