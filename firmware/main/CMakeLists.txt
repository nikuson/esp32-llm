idf_component_register(SRCS "main.c" "llm.c" "mv_pie.S"
                       INCLUDE_DIRS "."
                       PRIV_REQUIRES esp_partition esp_timer driver spi_flash esp_hw_support
                       LDFRAGMENTS "linker.lf")
target_compile_definitions(${COMPONENT_LIB} PRIVATE LLM_HAVE_PIE=1)
# keep float ops in exactly the order the reference uses (no fused multiply-add)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -ffp-contract=off)
