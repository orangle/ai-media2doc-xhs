from __future__ import annotations

from typing import List, TypedDict


class Fact(TypedDict):
    地点: str
    费用: str
    玩法: List[str]
    交通: str
    时间: str
    注意事项: List[str]
    标签: List[str]


class Post(TypedDict):
    title: str
    markdown: str


class PipelineResult(TypedDict, total=False):
    facts: Fact
    post: Post
