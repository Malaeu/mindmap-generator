PY ?= python

CERT_DIR := certs
REPORT_DIR := reports

FAMILIES := gaussian cauchy
GRID := phase3

.PHONY: certify-1k certify-10k certify-100k export-certificates five-pro

$(CERT_DIR):
	@mkdir -p $(CERT_DIR)

certify-1k: | $(CERT_DIR)
	@for F in $(FAMILIES); do \
		$(PY) -m experiments.certified_validation --family $$F --zeros 1000 --grid $(GRID) --p-cut 5000 --json $(CERT_DIR)/$${F}_1k.json ; \
	done

certify-10k: | $(CERT_DIR)
	@for F in $(FAMILIES); do \
		$(PY) -m experiments.certified_validation --family $$F --zeros 10000 --grid $(GRID) --p-cut 10000 --json $(CERT_DIR)/$${F}_10k.json ; \
	done

certify-100k: | $(CERT_DIR)
	@for F in $(FAMILIES); do \
		$(PY) -m experiments.certified_validation --family $$F --zeros 100000 --grid $(GRID) --p-cut 20000 --json $(CERT_DIR)/$${F}_100k.json ; \
	done

export-certificates: | $(REPORT_DIR)
	@$(PY) -m experiments.export_certificates $(REPORT_DIR)/CERTIFICATES.md

$(REPORT_DIR):
	@mkdir -p $(REPORT_DIR)

# Phase IV: Truncation certificate targets
.PHONY: truncate-cert
truncate-cert:
	@echo "Running truncation certificate (Gaussian grid, 100k zeros)..."
	@$(PY) -m experiments.weil_truncated --zeros=100000 --family gaussian --grid "0.6:1.4:0.02" --N-max 100000 --t-max 50 --plot

.PHONY: inverse-search
inverse-search:
	@echo "Running adversarial search (min Q_lower)..."
	@$(PY) -m experiments.inverse_h_search --zeros 100000 --family gaussian --sigma-grid "0.6:1.4:0.01" --heatmap

.PHONY: certificate
certificate: truncate-cert
	@echo "Truncation certificate run complete!"

# One-shot bundle: runs all families with strong params and zips artifacts
five-pro:
	@bash scripts/five_pro.sh "$(PY)"
