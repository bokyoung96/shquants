from etfs import paths
from etfs.families import build_parser as build_families_parser
from etfs.fnguide.coverage import build_parser as build_fnguide_coverage_parser
from etfs.fnguide.data_requirements import build_parser as build_data_requirements_parser
from etfs.fnguide.index_methodology import build_parser as build_index_methodology_parser
from etfs.fnguide.methodology import build_parser as build_fnguide_methodology_parser
from etfs.fnguide.methodology_audit import build_parser as build_methodology_audit_parser
from etfs.fnguide.methodology_engine import build_parser as build_methodology_engine_parser
from etfs.fnguide.methodology_extraction import build_parser as build_methodology_extraction_parser
from etfs.fnguide.methodology_specs import build_parser as build_methodology_specs_parser
from etfs.fnguide.pipeline import build_parser as build_fnguide_pipeline_parser
from etfs.krx.methodology import build_parser as build_krx_parser
from etfs.msci.methodology import build_parser as build_msci_parser
from etfs.nasdaq.methodology import build_parser as build_nasdaq_parser
from etfs.research import build_parser as build_research_parser
from etfs.sources import build_parser as build_sources_parser
from etfs.spglobal.methodology import build_parser as build_spglobal_parser


def test_output_path_constants_group_outputs_by_role() -> None:
    assert paths.UNIVERSE_OUTPUT_DIR.as_posix() == "etfs/output/universe"
    assert paths.CLASSIFICATION_OUTPUT_DIR.as_posix() == "etfs/output/classification"
    assert paths.SOURCES_OUTPUT_DIR.as_posix() == "etfs/output/sources"
    assert paths.FNGUIDE_OUTPUT_DIR.as_posix() == "etfs/output/providers/fnguide"
    assert paths.KRX_OUTPUT_DIR.as_posix() == "etfs/output/providers/krx"
    assert paths.SPGLOBAL_OUTPUT_DIR.as_posix() == "etfs/output/providers/spglobal"
    assert paths.NASDAQ_OUTPUT_DIR.as_posix() == "etfs/output/providers/nasdaq"
    assert paths.MSCI_OUTPUT_DIR.as_posix() == "etfs/output/providers/msci"
    assert paths.FNGUIDE_EXTRACTION_OUTPUT_DIR.as_posix() == "etfs/output/extractions/fnguide"
    assert paths.FNGUIDE_ENGINE_OUTPUT_DIR.as_posix() == "etfs/output/engine/fnguide"
    assert paths.FNGUIDE_METHODOLOGY_REVIEW_QUEUE_JSON.as_posix() == (
        "etfs/output/extractions/fnguide/methodology_review_queue.json"
    )
    assert paths.FNGUIDE_ENGINE_INPUT_REQUIREMENTS_JSON.as_posix() == (
        "etfs/output/engine/fnguide/engine_input_requirements.json"
    )
    assert paths.FNGUIDE_ENGINE_INPUT_TEMPLATE_JSON.as_posix() == (
        "etfs/output/engine/fnguide/engine_inputs.template.json"
    )
    assert paths.FNGUIDE_ENGINE_SUPPORT_MATRIX_JSON.as_posix() == (
        "etfs/output/engine/fnguide/engine_support_matrix.json"
    )
    assert paths.FNGUIDE_ENGINE_PROMOTION_CANDIDATES_JSON.as_posix() == (
        "etfs/output/engine/fnguide/engine_promotion_candidates.json"
    )
    assert paths.FNGUIDE_METHODOLOGY_REPLICATION_REPORT_JSON.as_posix() == (
        "etfs/output/engine/fnguide/methodology_replication_report.json"
    )
    assert paths.FNGUIDE_TARGET_WEIGHTS_JSON.as_posix() == "etfs/output/engine/fnguide/target_weights.json"
    assert paths.TARGET_WEIGHT_VALIDATION_JSON.as_posix() == "etfs/output/validation/target_weight_validation.json"


def test_fnguide_replication_paths_are_grouped_under_replication_output() -> None:
    assert paths.REPLICATION_OUTPUT_DIR.as_posix() == "etfs/output/replication"
    assert paths.FNGUIDE_REPLICATION_OUTPUT_DIR.as_posix() == "etfs/output/replication/fnguide"
    assert paths.FNGUIDE_KSS_REQUIREMENTS_JSON.as_posix() == "etfs/output/replication/fnguide/kss_data_requirements.json"
    assert paths.FNGUIDE_KSS_SELECTED_BUCKETS_JSON.as_posix() == "etfs/output/replication/fnguide/kss_selected_buckets.json"
    assert paths.FNGUIDE_KSS_TARGET_WEIGHTS_JSON.as_posix() == "etfs/output/replication/fnguide/kss_target_weights.json"
    assert paths.FNGUIDE_KSS_VALIDATION_JSON.as_posix() == "etfs/output/replication/fnguide/kss_replication_validation.json"
    assert paths.FNGUIDE_KSS_VALIDATION_MD.as_posix() == "etfs/output/replication/fnguide/kss_replication_validation.md"
    assert paths.FNGUIDE_DATA_INVENTORY_JSON.as_posix() == "etfs/output/replication/fnguide/data_inventory.json"
    assert paths.FNGUIDE_DATA_INVENTORY_MD.as_posix() == "etfs/output/replication/fnguide/data_inventory.md"
    assert paths.FNGUIDE_KSS_DATA_INVENTORY_JSON.as_posix() == "etfs/output/replication/fnguide/kss_data_inventory.json"
    assert paths.FNGUIDE_KSS_DATA_INVENTORY_MD.as_posix() == "etfs/output/replication/fnguide/kss_data_inventory.md"


def test_cli_defaults_write_to_grouped_output_folders() -> None:
    assert build_research_parser().parse_args([]).output_dir == paths.UNIVERSE_OUTPUT_DIR.as_posix()
    assert build_fnguide_methodology_parser().parse_args([]).output_dir == paths.FNGUIDE_OUTPUT_DIR.as_posix()
    assert build_families_parser().parse_args([]).output_dir == paths.CLASSIFICATION_OUTPUT_DIR.as_posix()
    assert build_sources_parser().parse_args([]).output_dir == paths.SOURCES_OUTPUT_DIR.as_posix()
    assert build_index_methodology_parser().parse_args([]).output_dir == paths.FNGUIDE_OUTPUT_DIR.as_posix()
    assert build_data_requirements_parser().parse_args([]).output_dir == paths.FNGUIDE_OUTPUT_DIR.as_posix()
    assert build_fnguide_coverage_parser().parse_args([]).output_dir == paths.FNGUIDE_OUTPUT_DIR.as_posix()
    assert build_methodology_extraction_parser().parse_args([]).output_dir == paths.FNGUIDE_EXTRACTION_OUTPUT_DIR.as_posix()
    assert build_methodology_specs_parser().parse_args([]).output_dir == paths.FNGUIDE_EXTRACTION_OUTPUT_DIR.as_posix()
    assert build_methodology_audit_parser().parse_args([]).output_dir == paths.FNGUIDE_EXTRACTION_OUTPUT_DIR.as_posix()
    assert build_methodology_engine_parser().parse_args([]).output_dir == paths.FNGUIDE_ENGINE_OUTPUT_DIR.as_posix()
    assert build_fnguide_pipeline_parser().parse_args([]).extraction_output_dir == (
        paths.FNGUIDE_EXTRACTION_OUTPUT_DIR.as_posix()
    )
    assert build_fnguide_pipeline_parser().parse_args([]).engine_output_dir == paths.FNGUIDE_ENGINE_OUTPUT_DIR.as_posix()
    assert build_krx_parser().parse_args([]).output_dir == paths.KRX_OUTPUT_DIR.as_posix()
    assert build_spglobal_parser().parse_args([]).output_dir == paths.SPGLOBAL_OUTPUT_DIR.as_posix()
    assert build_nasdaq_parser().parse_args([]).output_dir == paths.NASDAQ_OUTPUT_DIR.as_posix()
    assert build_msci_parser().parse_args([]).output_dir == paths.MSCI_OUTPUT_DIR.as_posix()


def test_cli_defaults_read_from_grouped_output_inputs() -> None:
    assert build_fnguide_methodology_parser().parse_args([]).input == paths.SECTOR_CSV.as_posix()
    assert build_families_parser().parse_args([]).universe == paths.UNIVERSE_JSON.as_posix()
    assert build_families_parser().parse_args([]).fnguide == paths.FNGUIDE_PDFS_JSON.as_posix()
    assert build_sources_parser().parse_args([]).families == paths.FAMILIES_JSON.as_posix()
    assert build_index_methodology_parser().parse_args([]).manifest == paths.FNGUIDE_PDFS_JSON.as_posix()
    assert build_data_requirements_parser().parse_args([]).rules == paths.FNGUIDE_RULES_JSON.as_posix()
    assert build_fnguide_coverage_parser().parse_args([]).requirements == paths.FNGUIDE_REQUIREMENTS_JSON.as_posix()
    assert build_methodology_extraction_parser().parse_args([]).rules == paths.FNGUIDE_RULES_JSON.as_posix()
    assert build_methodology_specs_parser().parse_args([]).extractions == paths.FNGUIDE_EXTRACTIONS_JSON.as_posix()
    assert build_methodology_audit_parser().parse_args([]).specs == paths.FNGUIDE_METHODOLOGY_SPECS_JSON.as_posix()
    assert build_methodology_engine_parser().parse_args([]).inputs == paths.FNGUIDE_ENGINE_INPUTS_JSON.as_posix()
    assert build_methodology_engine_parser().parse_args([]).specs == paths.FNGUIDE_METHODOLOGY_SPECS_JSON.as_posix()
    assert build_fnguide_pipeline_parser().parse_args([]).rules == paths.FNGUIDE_RULES_JSON.as_posix()
    assert build_fnguide_pipeline_parser().parse_args([]).overrides == paths.FNGUIDE_SPEC_OVERRIDES_JSON.as_posix()
    assert build_krx_parser().parse_args([]).sources == paths.SOURCES_JSON.as_posix()
    assert build_spglobal_parser().parse_args([]).sources == paths.SOURCES_JSON.as_posix()
    assert build_nasdaq_parser().parse_args([]).sources == paths.SOURCES_JSON.as_posix()
    assert build_msci_parser().parse_args([]).sources == paths.SOURCES_JSON.as_posix()
