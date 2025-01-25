# LLM Web Trainer Project - backend

**언어 선택 / Language Selection:**

- [🇰🇷 한국어 (Korean)](README.ko.md)

## Requirements

* [Poetry](https://python-poetry.org/) - Python 패키지 및 환경 관리

## 로컬 개발

### 일반적인 워크플로우

기본적으로 의존성은 [Poetry](https://python-poetry.org/)로 관리됩니다. Poetry를 설치한 후 아래 명령어로 의존성을 설치하세요.

poetry 설치 방법
```console
curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry
```

`./backend/` 폴더에서 모든 의존성을 설치하려면:

```console
$ poetry install --no-root
```

그 후 새로운 환경으로 셸 세션을 시작하려면:

```console
$ bash $(poetry env activate)
```
### pre-commit 설정
1. pre-commit을 설치하세요
```bash
apt install -y pre-commit
```
2. pre-commit을 적용하세요
```bash
pre-commit install
```

## 백엔드 테스트

백엔드를 테스트하려면:

```console
$ chmod +x ./scripts/test.sh
$ ./scripts/test.sh
```

테스트는 Pytest로 실행되며, `./backend/tests/`에서 테스트를 수정하거나 추가할 수 있습니다.

GitHub Actions를 사용하면 테스트가 자동으로 실행됩니다.

### 테스트 커버리지

테스트를 실행하면 `htmlcov/index.html` 파일이 생성됩니다. 이 파일을 브라우저에서 열어 테스트의 커버리지를 확인할 수 있습니다.