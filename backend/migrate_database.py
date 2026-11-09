
from sqlalchemy import inspect, text
from database import engine

new_columns = {
    "predicted_severity": "VARCHAR(50)",
    "prediction_confidence": "FLOAT",
    "verified_severity": "VARCHAR(50)",
    "verification_status": "VARCHAR(30) NOT NULL DEFAULT 'Pending'"
}

with engine.begin() as connection:

    inspector = inspect(connection)

    existing_columns = {
        column["name"]
        for column in inspector.get_columns("accidents")
    }

    for column_name, column_type in new_columns.items():

        if column_name not in existing_columns:

            connection.execute(
                text(
                    f"ALTER TABLE accidents "
                    f"ADD COLUMN {column_name} {column_type}"
                )
            )

            print(f"Added column: {column_name}")

        else:
            print(f"Already exists: {column_name}")

print("\nDatabase migration completed successfully!")
