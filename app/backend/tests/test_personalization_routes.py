import onnxruntime as ort

from tests.fakes import DIM, patch_base_data

VECTOR = {"dim": DIM, "indices": [0, 5], "values": [1.0, 5.0]}


def correction(message_id, label="Work", model="category"):
    return {
        "model": model,
        "provider_message_id": message_id,
        "feature_vector": VECTOR,
        "predicted_label": "Other",
        "predicted_confidence": 0.7,
        "corrected_label": label,
    }


def enable(client):
    assert client.put("/personalization/settings", json={"enabled": True}).json() == {"enabled": True}


def test_status_requires_a_session(client):
    assert client.get("/personalization/status").status_code == 401


def test_status_starts_empty_and_disabled(client, connected):
    body = client.get("/personalization/status").json()
    assert body["enabled"] is False
    assert [m["model"] for m in body["models"]] == ["category", "priority", "spam"]
    assert all(m["correction_count"] == 0 and m["corrections_until_retrain"] == 5 for m in body["models"])
    assert body["custom_labels"] == []


def test_corrections_are_refused_until_opted_in(client, connected, monkeypatch):
    patch_base_data(monkeypatch)
    assert client.post("/personalization/corrections", json=correction("m1")).status_code == 403


def test_recorrecting_an_email_replaces_the_earlier_correction(client, connected, monkeypatch):
    patch_base_data(monkeypatch)
    enable(client)
    client.post("/personalization/corrections", json=correction("m1", "Work"))
    response = client.post("/personalization/corrections", json=correction("m1", "Personal"))
    assert response.status_code == 200
    assert response.json()["corrections_until_retrain"] == 4
    assert client.get("/personalization/status").json()["models"][0]["correction_count"] == 1


def test_a_vector_of_the_wrong_dimension_is_rejected(client, connected, monkeypatch):
    patch_base_data(monkeypatch)
    enable(client)
    bad = correction("m1")
    bad["feature_vector"] = {"dim": DIM + 1, "indices": [0], "values": [1.0]}
    assert client.post("/personalization/corrections", json=bad).status_code == 422


def test_the_fifth_correction_retrains_and_publishes_a_model(client, connected, monkeypatch):
    patch_base_data(monkeypatch)
    enable(client)
    for i in range(4):
        body = client.post("/personalization/corrections", json=correction(f"m{i}", "Finance")).json()
        assert body["retrain_scheduled"] is False
    fifth = client.post("/personalization/corrections", json=correction("m4", "Finance")).json()
    assert fifth["retrain_scheduled"] is True

    # TestClient runs background tasks before returning, so the job is done.
    status = client.get("/personalization/status").json()
    assert status["models"][0]["last_attempt"]["status"] == "active"
    assert status["custom_labels"] == ["Finance"]
    manifest = client.get("/personalization/models").json()
    assert manifest == [{"model": "category", "version": 1, "classes": ["Finance", "Other", "Personal", "Work"]}]
    onnx_bytes = client.get("/personalization/models/category/1.onnx").content
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape[1] == DIM


def test_manual_retrain_only_schedules_models_with_new_corrections(client, connected, monkeypatch):
    patch_base_data(monkeypatch)
    enable(client)
    client.post("/personalization/corrections", json=correction("m1", "Work"))
    assert client.post("/personalization/retrain").json() == {"scheduled": ["category"]}
    assert client.post("/personalization/retrain").json() == {"scheduled": []}


def test_opting_out_deletes_corrections_and_models(client, connected, monkeypatch):
    patch_base_data(monkeypatch)
    enable(client)
    for i in range(5):
        client.post("/personalization/corrections", json=correction(f"m{i}", "Work"))
    assert client.get("/personalization/models").json() != []

    assert client.put("/personalization/settings", json={"enabled": False}).json() == {"enabled": False}

    status = client.get("/personalization/status").json()
    assert status["models"][0]["correction_count"] == 0
    assert status["models"][0]["last_attempt"] is None
    assert client.get("/personalization/models").json() == []


def test_unknown_model_version_is_404(client, connected):
    assert client.get("/personalization/models/category/9.onnx").status_code == 404
