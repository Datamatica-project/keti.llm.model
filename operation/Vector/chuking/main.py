#!/usr/bin/env python3
# main.py
"""
런처 스크립트: chunking.py / chunk_pipeline.py 를 안전하게 위임 실행

사용 예시:
  python main.py --mode length --input data/raw/ --output out/chunks --max-tokens 700 --overlap 100
  python main.py --mode semantic --input sample.txt --output out/chunks --model-name unsloth/gemma-3-4b-it
  python main.py --mode hybrid --input sample.txt --output out/chunks --cap-tokens 800 --overlap 120
"""

import os
import sys
import traceback


def _ensure_local_syspath():
    """현재 파일이 있는 디렉터리를 sys.path 맨 앞에 추가 (로컬 모듈 import 보장)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


def _import_pipeline_module():
    """
    chunking.py(우선) → chunk_pipeline.py(대체) 순으로 import 시도.
    반환: (module, name_str)
    """
    try:
        import chunking as pipeline  # 사용자가 제공한 파일명
        return pipeline, "chunking.py"
    except Exception as e1:
        # 대체 이름도 지원
        try:
            import chunk_pipeline as pipeline
            return pipeline, "chunk_pipeline.py"
        except Exception as e2:
            print("  파이프라인 모듈 import 실패:")
            print(" - chunking.py import 에러:", repr(e1))
            print(" - chunk_pipeline.py import 에러:", repr(e2))
            raise


def _print_header(pipeline_name: str):
    print("============================================================")
    print(f"  Chunking Launcher: using [{pipeline_name}]")
    print("    - 동일한 CLI 옵션을 그대로 지원합니다.")
    print("    - GemmaAgenticChunker.py가 있으면 semantic/hybrid 모드 사용 가능")
    print("============================================================")


def main():
    _ensure_local_syspath()

    try:
        pipeline, pipeline_name = _import_pipeline_module()
        _print_header(pipeline_name)

        # 가능하면 해당 모듈의 main(argv)을 직접 호출하여 기존 CLI 파서/로직을 그대로 사용
        if hasattr(pipeline, "main") and callable(pipeline.main):
            # 그대로 sys.argv를 전달 (argv=None로 내부 parse 하게)
            rc = pipeline.main(None)
            # 정수 반환이 아니면 0으로 간주
            try:
                rc = int(rc)
            except Exception:
                rc = 0
            return rc

        # 만약 main이 없다면 build_argparser를 찾아서 직접 파싱 후, 진입점 유사 구현
        elif hasattr(pipeline, "build_argparser") and callable(pipeline.build_argparser):
            parser = pipeline.build_argparser()
            args = parser.parse_args()  # sys.argv 사용
            # 모듈에 main이 없는데 build_argparser만 있다면,
            # 실제 처리 함수가 없다/구현 누락된 케이스이므로 에러 안내
            print(" 파이프라인 모듈에 main() 함수가 없습니다. main()을 구현하거나 제공 필요.")
            return 2

        else:
            print(" 파이프라인 모듈에 main()/build_argparser() 진입점 x.")
            return 2

    except SystemExit as e:
        # argparse가 내부적으로 SystemExit을 던질 수 있으니 그대로 exit code 전달
        return int(getattr(e, "code", 0))
    except KeyboardInterrupt:
        print("\n 사용자에 의해 중단되었습니다.")
        return 130
    except Exception:
        print(" 실행 중 예외가 발생했습니다:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
