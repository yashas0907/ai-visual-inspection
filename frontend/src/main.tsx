import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import './index.css'
import Dashboard from './pages/Dashboard'
import Inspect from './pages/Inspect'
import Result from './pages/Result'
import History from './pages/History'
import Analytics from './pages/Analytics'
import ModelInfo from './pages/ModelInfo'

function Shell() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <h1>SurfaceSpec</h1>
          <span>AI Visual Inspection</span>
        </div>
        <nav className="nav">
          <NavLink to="/" end>Dashboard</NavLink>
          <NavLink to="/inspect">New Inspection</NavLink>
          <NavLink to="/history">History</NavLink>
          <NavLink to="/analytics">Analytics</NavLink>
          <div className="nav-section">Platform</div>
          <NavLink to="/model">Model Info</NavLink>
        </nav>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/inspect" element={<Inspect />} />
          <Route path="/inspections/:id" element={<Result />} />
          <Route path="/history" element={<History />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/model" element={<ModelInfo />} />
        </Routes>
      </main>
    </div>
  )
}

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <BrowserRouter>
      <Shell />
    </BrowserRouter>
  </React.StrictMode>,
)
