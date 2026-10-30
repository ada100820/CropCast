#!/usr/bin/env Rscript
# =============================================================================
# run_epirice.R
# Fetch NASA POWER daily weather and run all 5 EPIRICE rice disease models.
#
# Usage:
#   Rscript run_epirice.R <lat> <lon> <establishment_date> <out_dir>
#
#   lat / lon            : decimal degrees  (e.g. 21.25  81.63)
#   establishment_date   : crop emergence,  YYYY-MM-DD  (e.g. 2023-07-15)
#   out_dir              : directory to write CSVs (created if absent)
#
# Example:
#   Rscript weather_risk/run_epirice.R 21.25 81.63 2023-07-15 /tmp/epirice_out
#
# -----------------------------------------------------------------------------
# INSTALL (R-universe build — use this, not CRAN, for the latest epicrop):
#
#   install.packages(
#     "epicrop",
#     repos = c(
#       "https://adamhsparks.r-universe.dev",
#       "https://cloud.r-project.org"
#     )
#   )
#
# Requires R >= 4.2.  NASA POWER queries need internet access.
# NOTE on R 4.6 / macOS Clang: R 4.6 defaults to -std=gnu23 which Apple Clang
#   does not accept.  Add to ~/.R/Makevars before installing packages:
#     CFLAGS   = -O2 -std=gnu2x
#     CXXFLAGS = -O2 -std=gnu++2a
# -----------------------------------------------------------------------------
# OUTPUT FILES (all in out_dir):
#   weather.csv           — raw daily weather (POWER data, one row per day)
#                           columns: YYYYMMDD, DOY, TEMP, TMIN, TMAX, RHUM, RAIN, LAT, LON
#   bacterial_blight.csv  — daily intensity, columns: dates, intensity
#   brown_spot.csv
#   leaf_blast.csv
#   sheath_blight.csv
#   tungro.csv
#
# NOTE ON EPICROP 1.0.0 API (verified 2026-10-30):
#   Disease functions are named  bacterial_blight(), brown_spot(), leaf_blast(),
#   sheath_blight(), tungro()  — NOT predict_*(). Output has 120 rows,
#   columns include "dates" (Date) and "intensity" (numeric 0..1).
#   "severity" column no longer exists in this version.
#   get_wth() signature: get_wth(lonlat, dates, duration=NULL, source=c("power","era5"))
# =============================================================================

# ---------- parse CLI args ---------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  cat("Usage: Rscript run_epirice.R <lat> <lon> <establishment_date> <out_dir>\n",
      file = stderr())
  quit(status = 1)
}

lat     <- as.numeric(args[1])
lon     <- as.numeric(args[2])
estab   <- args[3]          # "YYYY-MM-DD"
out_dir <- args[4]

if (is.na(lat) || is.na(lon)) {
  cat("ERROR: lat and lon must be numeric.\n", file = stderr())
  quit(status = 1)
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---------- load epicrop -----------------------------------------------------
if (!requireNamespace("epicrop", quietly = TRUE)) {
  stop(
    "epicrop is not installed.  Run in R:\n\n",
    "  install.packages('epicrop',\n",
    "    repos = c('https://adamhsparks.r-universe.dev',\n",
    "              'https://cloud.r-project.org'))\n"
  )
}
library(epicrop)

# ---------- fetch weather (NASA POWER) ---------------------------------------
# Season: establishment day through day+120.
# get_wth() signature:
#   get_wth(lonlat = c(longitude, latitude), dates = c("start", "end"))
# NOTE: lonlat takes LONGITUDE first, then LATITUDE.
start_date <- as.Date(estab)
end_date   <- start_date + 120

cat(sprintf("Fetching NASA POWER weather  lat=%.4f  lon=%.4f  %s to %s\n",
            lat, lon, format(start_date), format(end_date)))

wth <- get_wth(
  lonlat = c(lon, lat),
  dates  = c(format(start_date), format(end_date))
)
write.csv(wth, file = file.path(out_dir, "weather.csv"), row.names = FALSE)
cat("  weather.csv written\n")

# ---------- helper: extract intensity column ---------------------------------
# epicrop 1.0.0 uses "intensity"; guard against "severity" in older builds.
extract_intensity <- function(df, model_name) {
  cols <- names(df)
  if ("intensity" %in% cols) {
    return(df[["intensity"]])
  } else if ("severity" %in% cols) {
    return(df[["severity"]])
  } else {
    stop(sprintf(
      "Model '%s' output has neither 'intensity' nor 'severity' column.\n",
      model_name
    ), "Available columns: ", paste(cols, collapse = ", "))
  }
}

# ---------- helper: run one model and save CSV -------------------------------
# epicrop 1.0.0 function signatures (verified 2026-10-30):
#   bacterial_blight(wth, emergence, ...)
#   brown_spot(wth, emergence, ...)
#   leaf_blast(wth, emergence, ...)
#   sheath_blight(wth, emergence, ...)
#   tungro(wth, emergence, ...)
#
# 'emergence' is a character string "YYYY-MM-DD".
# Output: data.table, 120 rows, columns include "dates" (Date) and "intensity".
run_model <- function(name, fn) {
  cat(sprintf("  Running %s ...\n", name))
  result    <- fn(wth, emergence = estab)
  intensity <- extract_intensity(result, name)
  out <- data.frame(dates = result$dates, intensity = intensity)
  path <- file.path(out_dir, paste0(name, ".csv"))
  write.csv(out, file = path, row.names = FALSE)
  cat(sprintf("    -> %s  (rows=%d  max_intensity=%.4f)\n",
              path, nrow(out), max(intensity, na.rm = TRUE)))
  invisible(out)
}

# ---------- run all 5 EPIRICE models ----------------------------------------
cat("\nRunning EPIRICE models (establishment =", estab, ")\n")
run_model("bacterial_blight", function(w, emergence) bacterial_blight(w, emergence))
run_model("brown_spot",       function(w, emergence) brown_spot(w, emergence))
run_model("leaf_blast",       function(w, emergence) leaf_blast(w, emergence))
run_model("sheath_blight",    function(w, emergence) sheath_blight(w, emergence))
run_model("tungro",           function(w, emergence) tungro(w, emergence))

cat("\nDone. CSVs written to:", out_dir, "\n")
