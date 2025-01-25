#!/bin/sh
set -x

ruff check app scripts --fix
ruff format app scripts