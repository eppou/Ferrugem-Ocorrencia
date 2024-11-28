from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    dbstring: str

@dataclass
class Config:
    database_config: DatabaseConfig
