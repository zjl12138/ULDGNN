# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import zipfile
from typing import Dict, List

from .sketch_type import Contents, Document, Meta, SketchFile, Page, User, Workspace


# optional dataclass pretty printer
# using `pip install prettyprinter` to install
# import prettyprinter as pp


def from_file(file_path: str) -> SketchFile:
    """
    load SketchFile object from sketch file
    @param file_path: the sketch file path
    @return: the well structured SketchFile dataclass object
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'{file_path} not found')
    if not zipfile.is_zipfile(file_path):
        raise ValueError(f'{file_path} is not a zip file')
    zip_file = zipfile.ZipFile(file_path)
    document_dict = json.loads(zip_file.read('document.json'))
    if not (isinstance(document_dict, dict) and 'pages' in document_dict.keys()):
        raise ValueError(f'{file_path}/document.json is not valid')
    pages: List[Page] = []
    for page in document_dict['pages']:
        if not (isinstance(page, dict) and '_ref' in page.keys()):
            raise ValueError(f'{file_path}/document.json is not valid')
        page_dict = json.loads(zip_file.read(f"{page['_ref']}.json"))
        pages.append(page_dict)
    document_dict['pages'] = pages
    document: Document = Document.from_dict(document_dict)
    workspace: Workspace = {os.path.basename(file): json.loads(zip_file.read(file)) for file in
                            filter(
                                lambda x: x.startswith('workspace/') and x.endswith('.json'),
                                zip_file.namelist())}
    meta = Meta.from_dict(json.loads(zip_file.read('meta.json')))
    user: User = json.loads(zip_file.read('user.json'))
    return SketchFile(file_path, Contents(document, meta, user, workspace))


def to_file(obj: SketchFile, alter_file_path: str = None):
    """
    save changed SketchFile object to sketch file
    @param obj: SketchFile dataclass object
    @param alter_file_path: the new file_path
    """
    file_path = obj.filepath if alter_file_path is None else alter_file_path
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    zip_file = zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED)
    refs: List[Dict[str, str]] = []
    for page in obj.contents.document.pages:
        zip_file.writestr(
            os.path.join('pages', f'{page.do_objectID}.json'),
            json.dumps(page.to_dict(), ensure_ascii=False).encode('utf-8')
        )
        refs.append({
            '_class': 'MSJSONFileReference',
            '_ref_class': 'MSImmutablePage',
            '_ref': f'pages/{page.do_objectID}'
        })
    for (key, value) in obj.contents.workspace.items():
        zip_file.writestr(
            os.path.join('workspace', f'{key}.json'),
            json.dumps(value, ensure_ascii=False).encode('utf-8')
        )
    document_dict = obj.contents.document.to_dict()
    document_dict['pages'] = refs
    zip_file.writestr(
        'document.json',
        json.dumps(document_dict, ensure_ascii=False).encode('utf-8')
    )
    zip_file.writestr('user.json', json.dumps(
        obj.contents.user, ensure_ascii=False).encode('utf-8'))
    zip_file.writestr('meta.json', json.dumps(
        obj.contents.meta.to_dict(), ensure_ascii=False).encode('utf-8'))
    zip_file.close()


__all__ = [
    'from_file', 'to_file'
]

if __name__ == '__main__':
    pass
