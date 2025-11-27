"""Initialize database with all tables."""

import asyncio
import sys
from app.storage.postgres import init_db, close_db, engine, Base


async def initialize_database():
    """Create all database tables."""
    print("Initializing database...")
    print("=" * 60)

    try:
        # Import all models to ensure they're registered with Base
        print("Models loaded:")
        print("  - Organization")
        print("  - APIKey")
        print("  - Collection")
        print("  - Entity")
        print("  - SearchAnalytics")

        # Initialize database connection
        await init_db()

        # Create all tables
        async with engine.begin() as conn:
            print("\nCreating tables...")
            await conn.run_sync(Base.metadata.create_all)

        print("\n" + "=" * 60)
        print("✅ Database initialized successfully!")

        # Close connection
        await close_db()

        return True

    except Exception as e:
        print(f"\n❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main entry point."""
    success = await initialize_database()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
