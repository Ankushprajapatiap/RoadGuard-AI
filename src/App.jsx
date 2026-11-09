
import { useState, useEffect, useCallback } from 'react'
import 'leaflet/dist/leaflet.css'

import {
  MapContainer, TileLayer, CircleMarker, Popup
} from 'react-leaflet'

import {
  LayoutDashboard, Map, BrainCircuit, BarChart3,
  Settings, Bell, Search, Moon, Sun, Menu,
  Activity, Car, ShieldCheck, TrendingUp,
  AlertTriangle, Download, CheckCircle,
  MapPin, X, Plus, RefreshCw, ClipboardList,
  UserCheck, LogIn, LogOut, Lock, Filter,
  Clock, Database, AlertCircle
} from 'lucide-react'

import {
  BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

import { predictAndSaveIndianSeverity } from './api'

const API_BASE_URL = 'http://127.0.0.1:8000'

const SEVERITIES = [
  'Fatal',
  'Grievous Injury',
  'Minor Injury',
  'No Injury'
]

const NAV_ITEMS = [
  { name: 'Dashboard', icon: LayoutDashboard },
  { name: 'Add Accident', icon: Plus },
  { name: 'Accident Records', icon: ClipboardList },
  { name: 'Accident Map', icon: Map },
  { name: 'AI Prediction', icon: BrainCircuit },
  { name: 'Analytics', icon: BarChart3 },
  { name: 'Model Performance', icon: Activity },
  { name: 'Alerts', icon: Bell },
  { name: 'Officer Login', icon: Lock },
  { name: 'Settings', icon: Settings }
]

const DEFAULT_PREDICTION = {
  Day_of_Week: '1',
  Accident_Location_A: '1',
  Accident_Location_A_Chainage_km: '12.5',
  Accident_Location_A_Chainage_km_RoadSide: '1',
  Causes_D: '1',
  Road_Feature_E: '2',
  Road_Condition_F: '1',
  Weather_Conditions_H: '4',
  Vehicle_Type_Involved_J_V1: '1',
  Vehicle_Type_Involved_J_V2: 'Unknown',
  Year: '2023',
  Month: '6',
  Day: '15',
  Hour: '18',
  Minute: '30'
}

const styles = {
  page: { padding: '24px', color: 'inherit' },
  panel: {
    background: 'var(--panel, #182235)',
    border: '1px solid var(--border, #2b3850)',
    borderRadius: '14px',
    padding: '20px',
    marginBottom: '20px'
  },
  button: {
    border: 0,
    borderRadius: '8px',
    padding: '10px 14px',
    cursor: 'pointer',
    fontWeight: 600,
    display: 'inline-flex',
    alignItems: 'center',
    gap: '7px'
  },
  input: {
    width: '100%',
    padding: '11px',
    borderRadius: '8px',
    border: '1px solid var(--border, #40506a)',
    background: 'var(--input, #101a2b)',
    color: 'inherit',
    boxSizing: 'border-box'
  },
  label: {
    display: 'block',
    marginBottom: '7px',
    fontSize: '13px',
    fontWeight: 600
  }
}

function normaliseAccidents(data) {
  if (Array.isArray(data)) return data
  if (Array.isArray(data?.records)) return data.records
  if (Array.isArray(data?.items)) return data.items
  if (Array.isArray(data?.accidents)) return data.accidents
  return []
}

function formatDate(value) {
  if (!value) return 'Date unavailable'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value)
  return date.toLocaleString()
}

function getVerificationStatus(accident) {
  return accident.verification_status || 'Pending'
}

function getLocation(accident) {
  return accident.location ||
    accident.accident_location ||
    accident.Accident_Location_A ||
    'Unknown location'
}

function getPredictedSeverity(accident) {
  return accident.predicted_severity ||
    accident.predictedSeverity ||
    'Not predicted'
}

function getStatusColor(status) {
  const value = String(status || '').toLowerCase()

  if (value.includes('fatal') || value.includes('high') || value === 'rejected') {
    return '#ef4444'
  }

  if (value.includes('grievous') || value.includes('medium')) {
    return '#f59e0b'
  }

  if (value.includes('verified') || value.includes('no injury')) {
    return '#22c55e'
  }

  return '#94a3b8'
}

function StatCard({ title, value, subtitle, icon: Icon, color }) {
  return (
    <div className="stat-card" style={styles.panel}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <p style={{ opacity: 0.7, margin: '0 0 10px' }}>{title}</p>
          <h2 style={{ margin: 0, fontSize: '27px' }}>{value}</h2>
          <small style={{ opacity: 0.65 }}>{subtitle}</small>
        </div>
        <div style={{
          background: `${color}20`,
          color,
          padding: '13px',
          borderRadius: '12px'
        }}>
          <Icon size={25} />
        </div>
      </div>
    </div>
  )
}

function StatusBadge({ status }) {
  return (
    <span style={{
      color: getStatusColor(status),
      background: `${getStatusColor(status)}18`,
      padding: '5px 9px',
      borderRadius: '20px',
      fontSize: '12px',
      fontWeight: 700,
      whiteSpace: 'nowrap'
    }}>
      {status || 'Pending'}
    </span>
  )
}

function Field({ label, value, onChange, options, type = 'text' }) {
  return (
    <div className="form-field">
      <label style={styles.label}>{label}</label>
      {options ? (
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          style={styles.input}
        >
          {options.map(option => (
            <option key={option.value ?? option} value={option.value ?? option}>
              {option.label ?? option}
            </option>
          ))}
        </select>
      ) : (
        <input
          type={type}
          value={value}
          onChange={e => onChange(e.target.value)}
          style={styles.input}
        />
      )}
    </div>
  )
}

function AccidentMap({ accidents }) {
  const mapped = accidents.filter(accident => {
    const lat = Number(accident.latitude ?? accident.lat)
    const lng = Number(accident.longitude ?? accident.lng)
    return Number.isFinite(lat) && Number.isFinite(lng) &&
      Math.abs(lat) <= 90 && Math.abs(lng) <= 180
  })

  return (
    <div style={styles.panel}>
      <h3><MapPin size={19} /> Accident Locations</h3>
      {mapped.length === 0 ? (
        <p>No accident records with valid coordinates are available.</p>
      ) : (
        <MapContainer
          center={[26.9, 80.9]}
          zoom={7}
          style={{ height: '480px', width: '100%', borderRadius: '12px' }}
        >
          <TileLayer
            attribution='&copy; OpenStreetMap contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {mapped.map((accident, index) => (
            <CircleMarker
              key={accident.id ?? accident.accident_id ?? index}
              center={[
                Number(accident.latitude ?? accident.lat),
                Number(accident.longitude ?? accident.lng)
              ]}
              radius={9}
              pathOptions={{
                color: getStatusColor(
                  accident.predicted_severity || accident.severity
                ),
                fillOpacity: 0.75
              }}
            >
              <Popup>
                <strong>{getLocation(accident)}</strong>
                <br />
                Predicted: {getPredictedSeverity(accident)}
                <br />
                Status: {getVerificationStatus(accident)}
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      )}
      <p style={{ opacity: 0.65, fontSize: '12px' }}>
        Map markers are based on accident records returned by your backend.
      </p>
    </div>
  )
}

function AccidentRecords({
  accidents,
  loading,
  error,
  onRefresh,
  onVerify,
  onReject,
  officerLoggedIn,
  actionBusyId
}) {
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState('All')
  const [severitySelections, setSeveritySelections] = useState({})

  const filtered = accidents.filter(accident => {
    const location = getLocation(accident).toLowerCase()
    const id = String(accident.id ?? accident.accident_id ?? '').toLowerCase()
    const searchMatch =
      location.includes(search.toLowerCase()) ||
      id.includes(search.toLowerCase())

    const status = getVerificationStatus(accident)
    const statusMatch = statusFilter === 'All' || status === statusFilter

    return searchMatch && statusMatch
  })

  return (
    <div style={styles.panel}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        gap: '12px',
        flexWrap: 'wrap',
        alignItems: 'center'
      }}>
        <div>
          <h2 style={{ marginBottom: '5px' }}>Accident Records</h2>
          <p style={{ opacity: 0.7, marginTop: 0 }}>
            View, search and verify accident reports
          </p>
        </div>
        <button
          onClick={onRefresh}
          disabled={loading}
          style={{ ...styles.button, background: '#2563eb', color: 'white' }}
        >
          <RefreshCw size={16} /> {loading ? 'Loading...' : 'Refresh'}
        </button>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))',
        gap: '12px',
        margin: '18px 0'
      }}>
        <input
          placeholder="Search location or record ID..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={styles.input}
        />
        <select
          value={statusFilter}
          onChange={e => setStatusFilter(e.target.value)}
          style={styles.input}
        >
          <option>All</option>
          <option>Pending</option>
          <option>Verified</option>
          <option>Rejected</option>
        </select>
      </div>

      {!officerLoggedIn && (
        <div style={{
          padding: '12px',
          background: '#f59e0b18',
          border: '1px solid #f59e0b55',
          borderRadius: '9px',
          marginBottom: '15px'
        }}>
          <Lock size={16} style={{ verticalAlign: 'middle' }} /> Officer demo login
          is required to use Verify and Reject actions.
        </div>
      )}

      {error && (
        <div style={{
          padding: '12px',
          color: '#ef4444',
          background: '#ef444418',
          borderRadius: '8px',
          marginBottom: '15px'
        }}>
          <AlertCircle size={16} style={{ verticalAlign: 'middle' }} /> {error}
        </div>
      )}

      {loading ? (
        <p>Loading accident records from backend...</p>
      ) : filtered.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '35px 10px' }}>
          <Database size={35} opacity={0.5} />
          <h3>No accident records found</h3>
          <p style={{ opacity: 0.7 }}>
            Add an accident record or change your search/filter.
            If you expected records here, check that the backend is running.
          </p>
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            minWidth: '950px'
          }}>
            <thead>
              <tr style={{ textAlign: 'left', borderBottom: '1px solid #64748b55' }}>
                {[
                  'ID', 'Location', 'Road / Weather',
                  'Predicted Severity', 'Verification',
                  'Verified Severity', 'Date', 'Officer Action'
                ].map(item => (
                  <th key={item} style={{ padding: '12px 9px' }}>{item}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((accident, index) => {
                const id = accident.id ?? accident.accident_id
                const status = getVerificationStatus(accident)
                const selectedSeverity =
                  severitySelections[id] ??
                  (SEVERITIES.includes(getPredictedSeverity(accident))
                    ? getPredictedSeverity(accident)
                    : '')

                return (
                  <tr
                    key={id ?? index}
                    style={{ borderBottom: '1px solid #64748b30' }}
                  >
                    <td style={{ padding: '13px 9px' }}>{id ?? '—'}</td>
                    <td style={{ padding: '13px 9px' }}>
                      <strong>{getLocation(accident)}</strong>
                      <br />
                      <small style={{ opacity: 0.65 }}>
                        {accident.latitude ?? accident.lat ?? '—'},
                        {' '}
                        {accident.longitude ?? accident.lng ?? '—'}
                      </small>
                    </td>
                    <td style={{ padding: '13px 9px' }}>
                      {accident.road_type || accident.Road_Feature_E || '—'}
                      <br />
                      <small style={{ opacity: 0.65 }}>
                        {accident.weather || accident.Weather_Conditions_H || '—'}
                      </small>
                    </td>
                    <td style={{ padding: '13px 9px' }}>
                      <StatusBadge status={getPredictedSeverity(accident)} />
                    </td>
                    <td style={{ padding: '13px 9px' }}>
                      <StatusBadge status={status} />
                    </td>
                    <td style={{ padding: '13px 9px' }}>
                      {accident.verified_severity || '—'}
                    </td>
                    <td style={{ padding: '13px 9px' }}>
                      {formatDate(
                        accident.created_at || accident.date || accident.timestamp
                      )}
                    </td>
                    <td style={{ padding: '13px 9px' }}>
                      {status === 'Pending' ? (
                        <div style={{ minWidth: '170px' }}>
                          <select
                            value={selectedSeverity}
                            onChange={e => setSeveritySelections(previous => ({
                              ...previous,
                              [id]: e.target.value
                            }))}
                            style={{ ...styles.input, marginBottom: '7px' }}
                          >
                            <option value="">Select actual severity</option>
                            {SEVERITIES.map(severity => (
                              <option key={severity}>{severity}</option>
                            ))}
                          </select>
                          <div style={{ display: 'flex', gap: '5px' }}>
                            <button
                              disabled={!officerLoggedIn || !selectedSeverity || actionBusyId === id}
                              onClick={() => onVerify(id, selectedSeverity)}
                              style={{
                                ...styles.button,
                                background: '#16a34a',
                                color: 'white',
                                padding: '7px'
                              }}
                            >
                              <CheckCircle size={14} /> Verify
                            </button>
                            <button
                              disabled={!officerLoggedIn || actionBusyId === id}
                              onClick={() => onReject(id)}
                              style={{
                                ...styles.button,
                                background: '#dc2626',
                                color: 'white',
                                padding: '7px'
                              }}
                            >
                              <X size={14} /> Reject
                            </button>
                          </div>
                          {actionBusyId === id && <small>Updating...</small>}
                        </div>
                      ) : (
                        <small>Action completed</small>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      <p style={{ fontSize: '12px', opacity: 0.65, marginTop: '15px' }}>
        Showing {filtered.length} of {accidents.length} records.
        Verify/Reject actions call the backend verification endpoint.
      </p>
    </div>
  )
}

function AddAccidentForm({ onSave, saving, success, error }) {
  const [form, setForm] = useState({
    location: '',
    latitude: '26.8467',
    longitude: '80.9462',
    road_type: 'Highway',
    weather: 'Clear',
    traffic_density: 'Heavy',
    visibility: 'Good',
    accident_occurred: true,
    severity: 'Medium'
  })

  const update = (key, value) => {
    setForm(previous => ({ ...previous, [key]: value }))
  }

  const submit = e => {
    e.preventDefault()
    onSave({
      ...form,
      latitude: Number(form.latitude),
      longitude: Number(form.longitude),
      accident_occurred: Boolean(form.accident_occurred)
    })
  }

  return (
    <form onSubmit={submit} style={styles.panel}>
      <h2>Report a New Accident</h2>
      <p style={{ opacity: 0.7 }}>
        Enter accident details to save a new record in the backend database.
      </p>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit,minmax(210px,1fr))',
        gap: '16px'
      }}>
        <Field label="Location" value={form.location}
          onChange={v => update('location', v)} />
        <Field label="Latitude" type="number" value={form.latitude}
          onChange={v => update('latitude', v)} />
        <Field label="Longitude" type="number" value={form.longitude}
          onChange={v => update('longitude', v)} />
        <Field label="Road Type" value={form.road_type}
          onChange={v => update('road_type', v)}
          options={['Highway', 'Urban Road', 'Rural Road', 'Intersection']} />
        <Field label="Weather" value={form.weather}
          onChange={v => update('weather', v)}
          options={['Clear', 'Rainy', 'Foggy', 'Cloudy']} />
        <Field label="Traffic Density" value={form.traffic_density}
          onChange={v => update('traffic_density', v)}
          options={['Low', 'Moderate', 'Heavy']} />
        <Field label="Visibility" value={form.visibility}
          onChange={v => update('visibility', v)}
          options={['Good', 'Moderate', 'Poor']} />
        <Field label="Reported Severity" value={form.severity}
          onChange={v => update('severity', v)}
          options={['Low', 'Medium', 'High']} />
      </div>

      <label style={{
        display: 'flex',
        alignItems: 'center',
        gap: '9px',
        margin: '18px 0'
      }}>
        <input
          type="checkbox"
          checked={form.accident_occurred}
          onChange={e => update('accident_occurred', e.target.checked)}
        />
        Accident has occurred
      </label>

      {error && <p style={{ color: '#ef4444' }}>{error}</p>}
      {success && <p style={{ color: '#22c55e' }}>{success}</p>}

      <button
        type="submit"
        disabled={saving}
        style={{ ...styles.button, background: '#2563eb', color: 'white' }}
      >
        <Plus size={17} /> {saving ? 'Saving...' : 'Save Accident Record'}
      </button>
    </form>
  )
}

function AnalyticsPanel({ accidents, stats }) {
  const yearlyData = Array.isArray(stats?.yearly_accidents)
    ? stats.yearly_accidents
    : []

  const severityCounts = stats?.severity_counts || {}
  const predictedCounts = stats?.predicted_severity_counts || {}

  const severityData = SEVERITIES.map(severity => ({
    severity,
    recorded: Number(severityCounts[severity] || 0),
    predicted: Number(predictedCounts[severity] || 0)
  }))

  return (
    <>
      <div style={styles.panel}>
        <h2>Accident Analytics</h2>
        <p style={{ opacity: 0.7 }}>
          Charts use statistics returned by the backend.
        </p>
        <h3>Year-wise Accident Records</h3>
        {yearlyData.length === 0 ? (
          <p>No yearly analytics data available yet.</p>
        ) : (
          <div style={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={yearlyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      <div style={styles.panel}>
        <h3>Recorded vs Predicted Severity</h3>
        {accidents.length === 0 ? (
          <p>No accident data available for comparison.</p>
        ) : (
          <div style={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={severityData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="severity" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="recorded" fill="#2563eb" name="Recorded" />
                <Bar dataKey="predicted" fill="#f59e0b" name="Predicted" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </>
  )
}

function ModelPerformance() {
  const metrics = [
    { name: 'Accuracy', value: '45.99%' },
    { name: 'Balanced Accuracy', value: '43.72%' },
    { name: 'Macro F1 Score', value: '41.45%' },
    { name: 'Weighted F1 Score', value: '46.00%' }
  ]

  const classMetrics = [
    { name: 'Fatal', precision: '23%', recall: '20%', f1: '21%', support: 70 },
    { name: 'Grievous Injury', precision: '54%', recall: '47%', f1: '51%', support: 290 },
    { name: 'Minor Injury', precision: '52%', recall: '41%', f1: '46%', support: 309 },
    { name: 'No Injury', precision: '37%', recall: '66%', f1: '48%', support: 142 }
  ]

  return (
    <>
      <div style={styles.panel}>
        <h2><BrainCircuit size={22} /> ML Model Performance</h2>
        <p style={{ opacity: 0.7 }}>
          Indian accident severity classification model — last offline evaluation.
        </p>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit,minmax(170px,1fr))',
          gap: '13px'
        }}>
          {metrics.map(metric => (
            <div key={metric.name} style={{
              padding: '17px',
              borderRadius: '10px',
              background: '#64748b18'
            }}>
              <small style={{ opacity: 0.7 }}>{metric.name}</small>
              <h2 style={{ marginBottom: 0 }}>{metric.value}</h2>
            </div>
          ))}
        </div>

        <p style={{ marginTop: '18px' }}>
          <strong>Evaluation dataset:</strong> 3,242 training records and
          811 test records.
        </p>
      </div>

      <div style={styles.panel}>
        <h3>Class-wise Performance</h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: '500px' }}>
            <thead>
              <tr style={{ textAlign: 'left' }}>
                {['Class', 'Precision', 'Recall', 'F1 Score', 'Test Records'].map(item => (
                  <th key={item} style={{ padding: '12px' }}>{item}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {classMetrics.map(item => (
                <tr key={item.name} style={{ borderTop: '1px solid #64748b44' }}>
                  <td style={{ padding: '12px' }}>{item.name}</td>
                  <td style={{ padding: '12px' }}>{item.precision}</td>
                  <td style={{ padding: '12px' }}>{item.recall}</td>
                  <td style={{ padding: '12px' }}>{item.f1}</td>
                  <td style={{ padding: '12px' }}>{item.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div style={{
        ...styles.panel,
        border: '1px solid #f59e0b77',
        background: '#f59e0b12'
      }}>
        <h3><AlertTriangle size={19} /> Model Improvement Required</h3>
        <p>
          Current accuracy is 45.99%. Fatal-class recall is only 20%,
          so the model needs further evaluation before real-world use.
        </p>
        <ul>
          <li>Investigate class imbalance and test class weighting.</li>
          <li>Compare Random Forest with other suitable classifiers.</li>
          <li>Validate categorical encoding and input feature quality.</li>
          <li>Use cross-validation and tune hyperparameters.</li>
          <li>Evaluate recall and F1 for every severity class.</li>
        </ul>
        <small>
          These are improvement recommendations. This page does not retrain
          the model automatically.
        </small>
      </div>
    </>
  )
}

function OfficerLogin({ officerLoggedIn, onLogin, onLogout }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')

  const submit = e => {
    e.preventDefault()
    const result = onLogin(username, password)
    if (!result) setError('Invalid demo username or password.')
    else setError('')
  }

  if (officerLoggedIn) {
    return (
      <div style={styles.panel}>
        <h2><UserCheck size={22} /> Officer Session</h2>
        <StatusBadge status="Demo session active" />
        <p>You are logged in to the local frontend demonstration.</p>
        <button
          onClick={onLogout}
          style={{ ...styles.button, background: '#dc2626', color: 'white' }}
        >
          <LogOut size={16} /> Logout
        </button>
      </div>
    )
  }

  return (
    <div style={{ ...styles.panel, maxWidth: '520px', margin: '20px auto' }}>
      <div style={{ textAlign: 'center' }}>
        <ShieldCheck size={42} color="#3b82f6" />
        <h2>Officer Login</h2>
        <p style={{ opacity: 0.7 }}>RoadGuard AI officer demonstration</p>
      </div>

      <form onSubmit={submit}>
        <Field label="Username" value={username} onChange={setUsername} />
        <div style={{ marginTop: '15px' }}>
          <Field label="Password" type="password" value={password}
            onChange={setPassword} />
        </div>

        {error && <p style={{ color: '#ef4444' }}>{error}</p>}

        <button
          type="submit"
          style={{
            ...styles.button,
            width: '100%',
            justifyContent: 'center',
            background: '#2563eb',
            color: 'white',
            marginTop: '18px'
          }}
        >
          <LogIn size={17} /> Demo Login
        </button>
      </form>

      <div style={{
        background: '#f59e0b18',
        padding: '13px',
        borderRadius: '9px',
        marginTop: '20px',
        fontSize: '13px'
      }}>
        <strong><AlertTriangle size={15} /> Important security notice</strong>
        <p>
          This is a frontend-only demonstration login. Demo credentials are
          hardcoded in the frontend and are NOT secure authentication.
          Do not use this login for real government or production data.
        </p>
        <p>
          Production deployment requires backend authentication, password
          hashing, JWT/session validation, role-based access control and
          audit logging.
        </p>
      </div>
    </div>
  )
}

function SettingsPanel({ settings, setSettings, backendStatus, onRefresh }) {
  const toggle = key => {
    setSettings(previous => ({
      ...previous,
      [key]: !previous[key]
    }))
  }

  return (
    <div style={styles.panel}>
      <h2>System Settings</h2>
      <p>RoadGuard AI application preferences</p>

      {[
        ['notifications', 'Enable notifications'],
        ['emailAlerts', 'Email alerts preference'],
        ['autoRefresh', 'Automatic dashboard refresh']
      ].map(([key, label]) => (
        <label key={key} style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '15px 0',
          borderBottom: '1px solid #64748b33'
        }}>
          {label}
          <input
            type="checkbox"
            checked={settings[key]}
            onChange={() => toggle(key)}
          />
        </label>
      ))}

      <div style={{ marginTop: '20px' }}>
        <h3>Backend Connection</h3>
        <StatusBadge status={backendStatus ? 'Online' : 'Offline'} />
        <p style={{ opacity: 0.7 }}>
          API: {API_BASE_URL}
        </p>
        <button
          onClick={onRefresh}
          style={{ ...styles.button, background: '#2563eb', color: 'white' }}
        >
          <RefreshCw size={16} /> Test / Refresh Connection
        </button>
      </div>
    </div>
  )
}

function Dashboard({
  stats,
  accidents,
  loading,
  backendStatus,
  onRefresh,
  onNavigate
}) {
  const total = stats?.total_accidents ?? accidents.length
  const pending = stats?.pending_verification ?? accidents.filter(
    a => getVerificationStatus(a) === 'Pending'
  ).length
  const verified = stats?.verified_records ?? accidents.filter(
    a => getVerificationStatus(a) === 'Verified'
  ).length
  const rejected = stats?.rejected_records ?? accidents.filter(
    a => getVerificationStatus(a) === 'Rejected'
  ).length

  return (
    <>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '12px',
        marginBottom: '20px'
      }}>
        <div>
          <h2 style={{ marginBottom: '5px' }}>Dashboard Overview</h2>
          <p style={{ opacity: 0.7, margin: 0 }}>
            Live accident monitoring and road safety statistics
          </p>
        </div>
        <button
          onClick={onRefresh}
          disabled={loading}
          style={{ ...styles.button, background: '#2563eb', color: 'white' }}
        >
          <RefreshCw size={16} /> Refresh Data
        </button>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))',
        gap: '16px'
      }}>
        <StatCard title="Total Accident Records" value={total}
          subtitle="Backend records" icon={Car} color="#3b82f6" />
        <StatCard title="Pending Verification" value={pending}
          subtitle="Awaiting officer review" icon={Clock} color="#f59e0b" />
        <StatCard title="Verified Records" value={verified}
          subtitle="Officer verified" icon={CheckCircle} color="#22c55e" />
        <StatCard title="Rejected Records" value={rejected}
          subtitle="Rejected reports" icon={AlertTriangle} color="#ef4444" />
      </div>

      <div style={{
        ...styles.panel,
        marginTop: '20px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '12px'
      }}>
        <div>
          <h3 style={{ margin: '0 0 6px' }}>Backend Status</h3>
          <StatusBadge status={backendStatus ? 'Online' : 'Offline'} />
        </div>
        <button
          onClick={() => onNavigate('Accident Records')}
          style={{ ...styles.button, background: '#2563eb', color: 'white' }}
        >
          <ClipboardList size={16} /> View Accident Records
        </button>
      </div>

      <div style={styles.panel}>
        <h3>Recent Accident Records</h3>
        {accidents.length === 0 ? (
          <p>No accident records are currently available.</p>
        ) : (
          accidents.slice(0, 5).map((accident, index) => (
            <div key={accident.id ?? accident.accident_id ?? index} style={{
              display: 'flex',
              justifyContent: 'space-between',
              gap: '12px',
              padding: '12px 0',
              borderBottom: '1px solid #64748b33'
            }}>
              <div>
                <strong>{getLocation(accident)}</strong>
                <br />
                <small style={{ opacity: 0.65 }}>
                  {formatDate(accident.created_at || accident.date)}
                </small>
              </div>
              <StatusBadge status={getVerificationStatus(accident)} />
            </div>
          ))
        )}
      </div>
    </>
  )
}

export default function App() {
  const [darkMode, setDarkMode] = useState(true)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [activePage, setActivePage] = useState('Dashboard')
  const [search, setSearch] = useState('')
  const [notificationsOpen, setNotificationsOpen] = useState(false)

  const [accidents, setAccidents] = useState([])
  const [dashboardStats, setDashboardStats] = useState({})
  const [accidentLoading, setAccidentLoading] = useState(false)
  const [accidentError, setAccidentError] = useState('')
  const [backendStatus, setBackendStatus] = useState(false)

  const [saving, setSaving] = useState(false)
  const [saveSuccess, setSaveSuccess] = useState('')
  const [saveError, setSaveError] = useState('')
  const [actionBusyId, setActionBusyId] = useState(null)
  const [actionMessage, setActionMessage] = useState('')

  const [officerLoggedIn, setOfficerLoggedIn] = useState(false)

  const [prediction, setPrediction] = useState(DEFAULT_PREDICTION)
  const [predictionResult, setPredictionResult] = useState(null)
  const [predictionLoading, setPredictionLoading] = useState(false)
  const [predictionError, setPredictionError] = useState('')
  const [selectedAccidentId, setSelectedAccidentId] = useState('')

  const [settings, setSettings] = useState({
    notifications: true,
    emailAlerts: false,
    autoRefresh: true
  })

  const refreshAccidents = useCallback(async () => {
    setAccidentLoading(true)
    setAccidentError('')

    try {
      const [recordsResponse, statsResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/accidents`),
        fetch(`${API_BASE_URL}/dashboard/stats`)
      ])

      if (!recordsResponse.ok) {
        throw new Error(`Accident API returned ${recordsResponse.status}`)
      }

      const recordsData = await recordsResponse.json()
      const records = normaliseAccidents(recordsData)

      setAccidents(records)
      setBackendStatus(true)

      if (statsResponse.ok) {
        const statsData = await statsResponse.json()
        setDashboardStats(statsData)
      } else {
        setDashboardStats({})
      }
    } catch (error) {
      setBackendStatus(false)
      setAccidentError(
        `Could not load records. Check that FastAPI is running. ${error.message}`
      )
    } finally {
      setAccidentLoading(false)
    }
  }, [])

  useEffect(() => {
    refreshAccidents()
  }, [refreshAccidents])

  useEffect(() => {
    if (!settings.autoRefresh) return undefined
    const interval = setInterval(() => {
      refreshAccidents()
    }, 60000)
    return () => clearInterval(interval)
  }, [settings.autoRefresh, refreshAccidents])

  const saveAccident = async payload => {
    setSaving(true)
    setSaveSuccess('')
    setSaveError('')

    try {
      const response = await fetch(`${API_BASE_URL}/accidents`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      const data = await response.json().catch(() => ({}))

      if (!response.ok) {
        throw new Error(data.detail || `Server returned ${response.status}`)
      }

      setSaveSuccess('Accident record saved successfully.')
      await refreshAccidents()
      setActivePage('Accident Records')
    } catch (error) {
      setSaveError(error.message || 'Unable to save accident record.')
    } finally {
      setSaving(false)
    }
  }

  const verifyAccident = async (id, severity) => {
    if (!officerLoggedIn) {
      setActionMessage('Please log in as an officer first.')
      return
    }

    if (!id || !severity) {
      setActionMessage('Select a valid injury severity before verification.')
      return
    }

    setActionBusyId(id)
    setActionMessage('')

    try {
      const response = await fetch(
        `${API_BASE_URL}/accidents/${id}/verify`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            status: 'Verified',
            verified_severity: severity
          })
        }
      )

      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(data.detail || `Verification failed: ${response.status}`)
      }

      setActionMessage(`Record ${id} successfully verified.`)
      await refreshAccidents()
    } catch (error) {
      setActionMessage(error.message)
    } finally {
      setActionBusyId(null)
    }
  }

  const rejectAccident = async id => {
    if (!officerLoggedIn) {
      setActionMessage('Please log in as an officer first.')
      return
    }

    if (!window.confirm(`Are you sure you want to reject record ${id}?`)) return

    setActionBusyId(id)
    setActionMessage('')

    try {
      const response = await fetch(
        `${API_BASE_URL}/accidents/${id}/verify`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            status: 'Rejected',
            verified_severity: null
          })
        }
      )

      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(data.detail || `Rejection failed: ${response.status}`)
      }

      setActionMessage(`Record ${id} successfully rejected.`)
      await refreshAccidents()
    } catch (error) {
      setActionMessage(error.message)
    } finally {
      setActionBusyId(null)
    }
  }

  const handleOfficerLogin = (username, password) => {
    // DEMO ONLY. This is not real authentication.
    if (username === 'officer' && password === 'RoadGuard@123') {
      setOfficerLoggedIn(true)
      setActionMessage('Demo officer session started.')
      return true
    }
    return false
  }

  const exportCSV = () => {
    if (accidents.length === 0) {
      alert('No accident records available to export.')
      return
    }

    const columns = [
      'id', 'location', 'latitude', 'longitude',
      'road_type', 'weather', 'traffic_density',
      'visibility', 'severity', 'predicted_severity',
      'verification_status', 'verified_severity', 'created_at'
    ]

    const escapeCSV = value =>
      `"${String(value ?? '').replaceAll('"', '""')}"`

    const rows = accidents.map(accident => columns.map(key => {
      if (key === 'id') return escapeCSV(accident.id ?? accident.accident_id)
      if (key === 'location') return escapeCSV(getLocation(accident))
      if (key === 'verification_status') {
        return escapeCSV(getVerificationStatus(accident))
      }
      return escapeCSV(accident[key])
    }).join(','))

    const csv = [columns.join(','), ...rows].join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'RoadGuard_Accident_Records.csv'
    link.click()
    URL.revokeObjectURL(url)
  }

  const runPrediction = async e => {
    e.preventDefault()
    setPredictionLoading(true)
    setPredictionError('')
    setPredictionResult(null)

    try {
      const inputData = Object.fromEntries(
        Object.entries(prediction).map(([key, value]) => {
          if (key === 'Vehicle_Type_Involved_J_V2' && value === 'Unknown') {
            return [key, value]
          }
          return [key, Number(value)]
        })
      )

      const result = await predictAndSaveIndianSeverity(
        selectedAccidentId || null,
        inputData
      )

      setPredictionResult(result)
      await refreshAccidents()
    } catch (error) {
      setPredictionError(
        error.message || 'Prediction failed. Check the API and model endpoint.'
      )
    } finally {
      setPredictionLoading(false)
    }
  }

  const updatePrediction = (key, value) => {
    setPrediction(previous => ({ ...previous, [key]: value }))
  }

  const unreadCount = accidents.filter(accident =>
    getVerificationStatus(accident) === 'Pending'
  ).length

  const filteredNav = NAV_ITEMS.filter(item =>
    item.name.toLowerCase().includes(search.toLowerCase())
  )

  const themeStyle = {
    minHeight: '100vh',
    display: 'flex',
    background: darkMode ? '#0b1220' : '#f1f5f9',
    color: darkMode ? '#e5e7eb' : '#172033',
    '--panel': darkMode ? '#182235' : '#ffffff',
    '--border': darkMode ? '#2b3850' : '#dbe3ee',
    '--input': darkMode ? '#101a2b' : '#f8fafc'
  }

  const pageContent = () => {
    switch (activePage) {
      case 'Dashboard':
        return (
          <Dashboard
            stats={dashboardStats}
            accidents={accidents}
            loading={accidentLoading}
            backendStatus={backendStatus}
            onRefresh={refreshAccidents}
            onNavigate={setActivePage}
          />
        )

      case 'Add Accident':
        return (
          <AddAccidentForm
            onSave={saveAccident}
            saving={saving}
            success={saveSuccess}
            error={saveError}
          />
        )

      case 'Accident Records':
        return (
          <>
            {actionMessage && (
              <div style={{
                ...styles.panel,
                color: actionMessage.toLowerCase().includes('failed') ||
                  actionMessage.toLowerCase().includes('error')
                  ? '#ef4444' : 'inherit'
              }}>
                {actionMessage}
                <button
                  onClick={() => setActionMessage('')}
                  style={{ ...styles.button, marginLeft: '12px' }}
                >
                  <X size={14} /> Close
                </button>
              </div>
            )}
            <AccidentRecords
              accidents={accidents}
              loading={accidentLoading}
              error={accidentError}
              onRefresh={refreshAccidents}
              onVerify={verifyAccident}
              onReject={rejectAccident}
              officerLoggedIn={officerLoggedIn}
              actionBusyId={actionBusyId}
            />
            <div style={styles.panel}>
              <button
                onClick={exportCSV}
                style={{ ...styles.button, background: '#2563eb', color: 'white' }}
              >
                <Download size={16} /> Export Records CSV
              </button>
            </div>
          </>
        )

      case 'Accident Map':
        return <AccidentMap accidents={accidents} />

      case 'AI Prediction':
        return (
          <>
            <div style={styles.panel}>
              <h2><BrainCircuit size={22} /> Indian Accident Severity Prediction</h2>
              <p style={{ opacity: 0.7 }}>
                Predicts injury severity using accident-related input features.
                This is not a prediction of whether an accident will occur.
              </p>

              <form onSubmit={runPrediction}>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit,minmax(210px,1fr))',
                  gap: '15px'
                }}>
                  {Object.entries(prediction).map(([key, value]) => (
                    <Field
                      key={key}
                      label={key.replaceAll('_', ' ')}
                      value={value}
                      onChange={v => updatePrediction(key, v)}
                      options={key === 'Vehicle_Type_Involved_J_V2'
                        ? ['Unknown', '1', '2', '3', '4', '5']
                        : undefined}
                    />
                  ))}
                </div>

                <div style={{ marginTop: '18px' }}>
                  <label style={styles.label}>
                    Existing accident record ID (optional)
                  </label>
                  <input
                    value={selectedAccidentId}
                    onChange={e => setSelectedAccidentId(e.target.value)}
                    placeholder="Enter ID to attach prediction to a record"
                    style={styles.input}
                  />
                </div>

                {predictionError && (
                  <p style={{ color: '#ef4444' }}>{predictionError}</p>
                )}

                <button
                  type="submit"
                  disabled={predictionLoading}
                  style={{
                    ...styles.button,
                    background: '#2563eb',
                    color: 'white',
                    marginTop: '18px'
                  }}
                >
                  <BrainCircuit size={17} />
                  {predictionLoading ? 'Predicting...' : 'Run ML Prediction'}
                </button>
              </form>
            </div>

            {predictionResult && (
              <div style={styles.panel}>
                <h3>Prediction Result</h3>
                <h2 style={{
                  color: getStatusColor(
                    predictionResult.predicted_severity ||
                    predictionResult.severity
                  )
                }}>
                  {predictionResult.predicted_severity ||
                    predictionResult.severity ||
                    predictionResult.prediction ||
                    'Result received'}
                </h2>
                <pre style={{
                  whiteSpace: 'pre-wrap',
                  overflowWrap: 'anywhere'
                }}>
                  {JSON.stringify(predictionResult, null, 2)}
                </pre>
              </div>
            )}
          </>
        )

      case 'Analytics':
        return <AnalyticsPanel accidents={accidents} stats={dashboardStats} />

      case 'Model Performance':
        return <ModelPerformance />

      case 'Alerts':
        return (
          <div style={styles.panel}>
            <h2><Bell size={22} /> Safety Alerts</h2>
            <p style={{ opacity: 0.7 }}>
              Alerts are generated from actual accident records.
            </p>
            {accidents.length === 0 ? (
              <p>No accident-based alerts available.</p>
            ) : accidents
              .filter(accident => {
                const severity = String(
                  accident.predicted_severity || accident.severity || ''
                ).toLowerCase()
                return severity.includes('fatal') ||
                  severity.includes('high') ||
                  getVerificationStatus(accident) === 'Pending'
              })
              .map((accident, index) => (
                <div key={accident.id ?? accident.accident_id ?? index} style={{
                  padding: '14px',
                  borderBottom: '1px solid #64748b33',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px'
                }}>
                  <AlertTriangle color="#f59e0b" />
                  <div>
                    <strong>{getLocation(accident)}</strong>
                    <p style={{ margin: '5px 0', opacity: 0.7 }}>
                      Predicted: {getPredictedSeverity(accident)}
                      {' · '}
                      Verification: {getVerificationStatus(accident)}
                    </p>
                  </div>
                </div>
              ))}
          </div>
        )

      case 'Officer Login':
        return (
          <OfficerLogin
            officerLoggedIn={officerLoggedIn}
            onLogin={handleOfficerLogin}
            onLogout={() => {
              setOfficerLoggedIn(false)
              setActionMessage('Demo officer logged out.')
            }}
          />
        )

      case 'Settings':
        return (
          <SettingsPanel
            settings={settings}
            setSettings={setSettings}
            backendStatus={backendStatus}
            onRefresh={refreshAccidents}
          />
        )

      default:
        return <Dashboard
          stats={dashboardStats}
          accidents={accidents}
          loading={accidentLoading}
          backendStatus={backendStatus}
          onRefresh={refreshAccidents}
          onNavigate={setActivePage}
        />
    }
  }

  return (
    <div className={`app ${darkMode ? 'dark' : 'light'}`} style={themeStyle}>
      {sidebarOpen && (
        <aside style={{
          width: '245px',
          flexShrink: 0,
          background: darkMode ? '#111b2d' : '#ffffff',
          borderRight: '1px solid var(--border)',
          padding: '20px 13px',
          boxSizing: 'border-box'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            padding: '5px 8px 22px'
          }}>
            <ShieldCheck size={30} color="#3b82f6" />
            <div>
              <strong style={{ fontSize: '18px' }}>RoadGuard AI</strong>
              <small style={{ display: 'block', opacity: 0.6 }}>
                Road Safety Platform
              </small>
            </div>
          </div>

          <input
            placeholder="Search menu..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{ ...styles.input, marginBottom: '15px' }}
          />

          {filteredNav.map(item => {
            const Icon = item.icon
            const active = activePage === item.name

            return (
              <button
                key={item.name}
                onClick={() => setActivePage(item.name)}
                style={{
                  width: '100%',
                  border: 0,
                  borderRadius: '9px',
                  padding: '12px',
                  marginBottom: '5px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '11px',
                  cursor: 'pointer',
                  textAlign: 'left',
                  background: active ? '#2563eb' : 'transparent',
                  color: active ? 'white' : 'inherit'
                }}
              >
                <Icon size={18} />
                {item.name}
                {item.name === 'Accident Records' && unreadCount > 0 && (
                  <span style={{
                    marginLeft: 'auto',
                    background: '#f59e0b',
                    color: '#111827',
                    padding: '2px 7px',
                    borderRadius: '20px',
                    fontSize: '11px'
                  }}>
                    {unreadCount}
                  </span>
                )}
              </button>
            )
          })}

          <div style={{
            marginTop: '25px',
            padding: '12px',
            background: '#22c55e12',
            borderRadius: '10px'
          }}>
            <small style={{ opacity: 0.7 }}>API CONNECTION</small>
            <p style={{ margin: '7px 0 0', color: backendStatus ? '#22c55e' : '#ef4444' }}>
              ● {backendStatus ? 'Backend Online' : 'Backend Offline'}
            </p>
          </div>
        </aside>
      )}

      <main style={{ flex: 1, minWidth: 0 }}>
        <header style={{
          height: '68px',
          padding: '0 24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: '1px solid var(--border)',
          background: darkMode ? '#111b2d' : '#ffffff',
          gap: '12px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              style={{ ...styles.button, background: 'transparent', padding: '6px' }}
              title="Toggle sidebar"
            >
              <Menu size={21} />
            </button>
            <strong>{activePage}</strong>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <button
              onClick={() => setDarkMode(!darkMode)}
              style={{ ...styles.button, background: 'transparent', padding: '7px' }}
              title="Toggle theme"
            >
              {darkMode ? <Sun size={19} /> : <Moon size={19} />}
            </button>

            <div style={{ position: 'relative' }}>
              <button
                onClick={() => setNotificationsOpen(!notificationsOpen)}
                style={{ ...styles.button, background: 'transparent', padding: '7px' }}
                title="Notifications"
              >
                <Bell size={19} />
                {unreadCount > 0 && (
                  <span style={{
                    background: '#ef4444',
                    color: 'white',
                    borderRadius: '50%',
                    padding: '2px 5px',
                    fontSize: '10px'
                  }}>
                    {unreadCount}
                  </span>
                )}
              </button>

              {notificationsOpen && (
                <div style={{
                  position: 'absolute',
                  right: 0,
                  top: '42px',
                  width: '260px',
                  zIndex: 1000,
                  padding: '15px',
                  background: darkMode ? '#182235' : 'white',
                  border: '1px solid var(--border)',
                  borderRadius: '10px',
                  boxShadow: '0 8px 25px #0003'
                }}>
                  <strong>Notifications</strong>
                  <p style={{ fontSize: '13px', opacity: 0.7 }}>
                    {unreadCount} records awaiting officer verification.
                  </p>
                  <button
                    onClick={() => {
                      setActivePage('Accident Records')
                      setNotificationsOpen(false)
                    }}
                    style={{
                      ...styles.button,
                      background: '#2563eb',
                      color: 'white'
                    }}
                  >
                    Review Records
                  </button>
                </div>
              )}
            </div>

            <div style={{
              width: '34px',
              height: '34px',
              borderRadius: '50%',
              background: '#2563eb',
              color: 'white',
              display: 'grid',
              placeItems: 'center',
              fontWeight: 700
            }}>
              AP
            </div>
          </div>
        </header>

        <div style={styles.page}>
          {pageContent()}

          <footer style={{
            textAlign: 'center',
            padding: '25px 10px',
            opacity: 0.55,
            fontSize: '12px'
          }}>
            RoadGuard AI · Traffic Accident Analysis & Road Safety Platform
            <br />
            Academic / placement demonstration — not for emergency response
            or operational government decisions.
          </footer>
        </div>
      </main>
    </div>
  )
}
