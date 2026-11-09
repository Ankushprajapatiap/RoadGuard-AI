
const API_BASE_URL = "http://127.0.0.1:8000";

// Normal Indian ML prediction
export async function predictIndianSeverity(inputData) {
  const response = await fetch(`${API_BASE_URL}/predict/india`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(inputData),
  });

  const result = await response.json();

  if (!response.ok) {
    throw new Error(
      typeof result.detail === "string"
        ? result.detail
        : "Indian ML prediction failed"
    );
  }

  return result;
}

// Generate and save prediction for an existing accident
export async function predictAndSaveIndianSeverity(accidentId, inputData) {
  const response = await fetch(
    `${API_BASE_URL}/accidents/${accidentId}/predict/india`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(inputData),
    }
  );

  const result = await response.json();

  if (!response.ok) {
    throw new Error(
      typeof result.detail === "string"
        ? result.detail
        : "Prediction could not be saved"
    );
  }

  return result;
}

// Fetch all accident records
export async function getAccidents() {
  const response = await fetch(`${API_BASE_URL}/accidents`);
  const result = await response.json();

  if (!response.ok) {
    throw new Error("Could not fetch accident records");
  }

  return result;
}

// Fetch dashboard statistics
export async function getDashboardStats() {
  const response = await fetch(`${API_BASE_URL}/dashboard/stats`);
  const result = await response.json();

  if (!response.ok) {
    throw new Error("Could not fetch dashboard statistics");
  }

  return result;
}
