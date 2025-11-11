"""Run database migrations."""

import asyncio
import sys
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine
from app.config import settings


async def run_migration():
    """Run SQL migration script."""
    migration_file = Path(__file__).parent / "migrations" / "001_initial_schema.sql"

    if not migration_file.exists():
        print(f"❌ Migration file not found: {migration_file}")
        return False

    # Read migration SQL
    with open(migration_file, "r") as f:
        sql = f.read()

    print(f"Running migration: {migration_file.name}")
    print("=" * 60)

    # Create engine
    engine = create_async_engine(settings.database_url, echo=True)

    try:
        async with engine.begin() as conn:
            # Split by semicolon and execute each statement
            statements = [s.strip() for s in sql.split(";") if s.strip()]

            for i, statement in enumerate(statements, 1):
                if statement:
                    print(f"\n[{i}/{len(statements)}] Executing...")
                    await conn.execute(statement)

        print("\n" + "=" * 60)
        print("✅ Migration completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        return False

    finally:
        await engine.dispose()


async def main():
    """Main entry point."""
    success = await run_migration()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
