#!/bin/sh
set -x

ruff check . scripts --fix
ruff format . scripts