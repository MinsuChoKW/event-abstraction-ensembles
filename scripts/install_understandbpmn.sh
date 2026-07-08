#!/usr/bin/env bash

set -euo pipefail

echo "============================================================"
echo "[Install] R and understandBPMN 1.1.1"
echo "============================================================"

# ------------------------------------------------------------
# 1. System dependencies
# ------------------------------------------------------------
if ! command -v Rscript >/dev/null 2>&1; then
    echo "[Install] R is not installed. Installing R..."

    apt-get -y update

    DEBIAN_FRONTEND=noninteractive apt-get -y install \
        r-base \
        r-base-dev \
        build-essential \
        gfortran \
        libxml2-dev \
        libcurl4-openssl-dev \
        libssl-dev \
        zlib1g-dev \
        wget \
        ca-certificates
else
    echo "[Install] R is already installed."
fi

echo "[Install] R version:"
Rscript --version

# ------------------------------------------------------------
# 2. Helper: install an R package only when missing
# ------------------------------------------------------------
install_r_package_if_missing() {
    local package_name="$1"

    if Rscript -e "quit(status=ifelse(requireNamespace('${package_name}', quietly=TRUE), 0, 1))"; then
        echo "[Install] R package already installed: ${package_name}"
    else
        echo "[Install] Installing R package: ${package_name}"

        Rscript -e "
            install.packages(
                '${package_name}',
                repos='https://cloud.r-project.org',
                dependencies=TRUE
            )
        "
    fi
}

# ------------------------------------------------------------
# 3. understandBPMN dependencies
# ------------------------------------------------------------
install_r_package_if_missing "jsonlite"
install_r_package_if_missing "Rcpp"
install_r_package_if_missing "XML"
install_r_package_if_missing "dplyr"
install_r_package_if_missing "purrr"
install_r_package_if_missing "tidyr"
install_r_package_if_missing "tibble"
install_r_package_if_missing "R.utils"

# ------------------------------------------------------------
# 4. Install understandBPMN 1.1.1 from CRAN Archive
# ------------------------------------------------------------
if Rscript -e "quit(status=ifelse(requireNamespace('understandBPMN', quietly=TRUE), 0, 1))"; then
    echo "[Install] understandBPMN is already installed."
else
    echo "[Install] Installing understandBPMN 1.1.1..."

    Rscript - <<'RSCRIPT'
url <- paste0(
    "https://cran.r-project.org/src/contrib/Archive/",
    "understandBPMN/understandBPMN_1.1.1.tar.gz"
)

dest <- "/tmp/understandBPMN_1.1.1.tar.gz"

download.file(
    url,
    dest,
    mode = "wb"
)

install.packages(
    dest,
    repos = NULL,
    type = "source"
)
RSCRIPT
fi

# ------------------------------------------------------------
# 5. Final verification
# ------------------------------------------------------------
echo "[Install] Verifying installation..."

Rscript - <<'RSCRIPT'
library(understandBPMN)
library(jsonlite)

cat("understandBPMN installation: OK\n")
cat("jsonlite installation: OK\n")
RSCRIPT

echo "============================================================"
echo "[Install] Done"
echo "============================================================"
