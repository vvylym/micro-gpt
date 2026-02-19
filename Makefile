SHELL := /bin/bash

CARGO ?= cargo
MAKE ?= make

.PHONY: check fmt fmt-check clippy test deny doc machete coverage coverage-check pmat ci clean run

check:
	$(CARGO) check --workspace --all-features --all-targets

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

clippy:
	$(CARGO) clippy --workspace --all-features --all-targets -- -D warnings

test:
	$(CARGO) test --workspace --all-features --all-targets

doc:
	$(CARGO) doc --workspace --all-features --no-deps

deny:
	$(CARGO) deny check

clean:
	$(CARGO) clean

run:
	$(CARGO) run -p microgpt-original

machete:
	@if ! $(CARGO) --list | grep -q 'machete'; then \
		echo "Installing cargo-machete..."; \
		$(CARGO) install cargo-machete --locked; \
	fi
	$(CARGO) machete --with-metadata

coverage:
	@if ! $(CARGO) --list | grep -q 'llvm-cov'; then \
		echo "Installing cargo-llvm-cov..."; \
		$(CARGO) install cargo-llvm-cov; \
	fi
	$(CARGO) llvm-cov --workspace --all-features --lcov --output-path lcov.info

coverage-check:
	@if ! $(CARGO) --list | grep -q 'llvm-cov'; then \
		echo "Installing cargo-llvm-cov..."; \
		$(CARGO) install cargo-llvm-cov; \
	fi
	$(CARGO) llvm-cov test --workspace --all-features --no-report
	$(CARGO) llvm-cov report --fail-under-lines 95

pmat:
	@if ! command -v pmat >/dev/null 2>&1; then \
		echo "Installing pmat..."; \
		$(CARGO) install pmat; \
	fi
	pmat quality-gate --checks dead-code,complexity,satd,security,duplicates,coverage --max-dead-code 45 --fail-on-violation

ci:
	$(MAKE) fmt-check
	$(MAKE) clippy
	$(MAKE) test
	$(MAKE) doc
	$(MAKE) deny
	$(MAKE) machete
	$(MAKE) coverage-check
	$(MAKE) pmat