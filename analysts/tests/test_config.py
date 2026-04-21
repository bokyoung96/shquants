from pathlib import Path

from analysts.config import build_config


def test_build_config_loads_gmail_settings(tmp_path: Path) -> None:
    (tmp_path / "config.local.json").write_text(
        """
        {
          "gmail": {
            "account_name": "reports-primary",
            "client_secret_path": "secrets/gmail-client.json",
            "token_path": "secrets/gmail-token.json",
            "query": "label:broker-reports newer_than:14d",
            "body_candidate_rules": {"min_chars": 800, "require_structure": true},
            "zip_allow_extensions": [".pdf", ".txt", ".html"],
            "poll_interval_seconds": 300
          }
        }
        """.strip()
    )

    config = build_config(tmp_path)

    assert config.gmail is not None
    assert config.gmail.account_name == "reports-primary"
    assert config.gmail.query == "label:broker-reports newer_than:14d"
    assert config.gmail.body_candidate_rules.min_chars == 800
    assert config.gmail.zip_allow_extensions == (".pdf", ".txt", ".html")


def test_build_config_loads_inline_gmail_client_secret_json(tmp_path: Path) -> None:
    (tmp_path / "config.local.json").write_text(
        """
        {
          "gmail": {
            "account_name": "shquants@gmail.com",
            "client_secret_json": {
              "installed": {
                "client_id": "abc.apps.googleusercontent.com",
                "project_id": "demo-project",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_secret": "secret",
                "redirect_uris": ["http://localhost"]
              }
            },
            "token_path": "gmail-token.json",
            "query": ""
          }
        }
        """.strip()
    )

    config = build_config(tmp_path)

    assert config.gmail is not None
    assert config.gmail.account_name == "shquants@gmail.com"
    assert config.gmail.client_secret_path is None
    assert config.gmail.client_secret_json is not None
    assert config.gmail.client_secret_json["installed"]["project_id"] == "demo-project"


def test_build_config_creates_source_specific_raw_dirs(tmp_path: Path) -> None:
    config = build_config(tmp_path)

    assert config.paths.raw_dir == tmp_path / "data" / "raw"
    assert config.paths.telegram_raw_dir == tmp_path / "data" / "raw" / "telegram"
    assert config.paths.gmail_raw_dir == tmp_path / "data" / "raw" / "gmail"
