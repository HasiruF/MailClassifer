import pytest
from sqlalchemy.exc import IntegrityError

from src.models import Correction, CorrectionModel, User


def test_personalization_is_off_by_default(db):
    user = User()
    db.add(user)
    db.commit()
    db.refresh(user)
    assert user.personalization_enabled is False


def _correction(connection_id, label):
    return Correction(
        email_connection_id=connection_id,
        provider_message_id="msg-1",
        model=CorrectionModel.category,
        feature_vector={"dim": 3, "indices": [0], "values": [1.0]},
        predicted_label="Other",
        predicted_confidence=0.6,
        corrected_label=label,
    )


def test_one_correction_per_message_and_model(db, connected):
    db.add(_correction(connected.connection.id, "Work"))
    db.commit()
    db.add(_correction(connected.connection.id, "Personal"))
    with pytest.raises(IntegrityError):
        db.commit()
