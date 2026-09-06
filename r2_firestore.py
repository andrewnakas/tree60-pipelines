"""
Drop-in replacement for the slice of `firebase_admin.firestore` the Tree 60
pipelines use, backed by JSON objects in Cloudflare R2.

These pipelines use Firestore only as a small key/value store for run manifests
and pointers — document set/get/delete, a subcollection of frames, and one
unfiltered stream over a collection. None of that needs a database, and every
document written here was costing Firestore writes on every render pass.

    from firebase_admin import firestore as fb_firestore   ->  import r2_firestore as fb_firestore
    fb_firestore.client()                                  ->  unchanged

Layout in R2, so a document and its subcollections can coexist:

    _fs/<collection>/<doc>.json
    _fs/<collection>/<doc>/<subcollection>/<subdoc>.json

Not supported (and deliberately loud about it): where() filters, order_by,
transactions, and server-side timestamps. Add them only if a ported module
actually needs them.
"""

import json
import logging
import posixpath

import r2_storage

logger = logging.getLogger(__name__)

ROOT = "_fs"


def _bucket():
    return r2_storage.Client().bucket()


class DocumentSnapshot:
    def __init__(self, doc_id, data, reference=None):
        self.id = doc_id
        self._data = data
        self.reference = reference

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return dict(self._data) if self._data is not None else None

    def get(self, key, default=None):
        return (self._data or {}).get(key, default)


class DocumentReference:
    def __init__(self, path, doc_id):
        self._path = path           # _fs/<collection>/<doc>
        self.id = doc_id

    @property
    def _key(self):
        return f"{self._path}.json"

    def collection(self, name):
        """Subcollection lives under the document's own path."""
        return CollectionReference(posixpath.join(self._path, name))

    def get(self):
        blob = _bucket().blob(self._key)
        if not blob.exists():
            return DocumentSnapshot(self.id, None, self)
        try:
            return DocumentSnapshot(self.id, json.loads(blob.download_as_text()), self)
        except Exception as e:
            logger.warning("unreadable document %s: %s", self._key, e)
            return DocumentSnapshot(self.id, None, self)

    def set(self, data, merge=False):
        payload = dict(data)
        if merge:
            existing = self.get().to_dict() or {}
            existing.update(payload)
            payload = existing
        blob = _bucket().blob(self._key)
        blob.upload_from_string(
            json.dumps(payload, default=str, separators=(",", ":")),
            content_type="application/json",
        )
        return self

    def update(self, data):
        return self.set(data, merge=True)

    def delete(self):
        blob = _bucket().blob(self._key)
        if blob.exists():
            blob.delete()


class CollectionReference:
    def __init__(self, path):
        self._path = path           # _fs/<collection>

    def document(self, doc_id=None):
        if doc_id is None:
            import uuid

            doc_id = uuid.uuid4().hex
        return DocumentReference(posixpath.join(self._path, doc_id), doc_id)

    def stream(self):
        """
        Documents directly in this collection. Keys one level deeper belong to a
        subcollection and are skipped, matching Firestore's behaviour.
        """
        prefix = self._path + "/"
        for blob in _bucket().list_blobs(prefix=prefix):
            rest = blob.name[len(prefix):]
            if not rest.endswith(".json") or "/" in rest:
                continue
            doc_id = rest[: -len(".json")]
            yield DocumentReference(posixpath.join(self._path, doc_id), doc_id).get()

    # Firestore's fluent API is used in a couple of spots; keep get() as an alias.
    def get(self):
        return list(self.stream())

    def limit(self, n):
        return _LimitedCollection(self, n)


class _LimitedCollection:
    def __init__(self, coll, n):
        self._coll = coll
        self._n = n

    def stream(self):
        for i, doc in enumerate(self._coll.stream()):
            if i >= self._n:
                return
            yield doc


class Client:
    def collection(self, name):
        return CollectionReference(posixpath.join(ROOT, name))

    def document(self, path):
        parts = path.strip("/").split("/")
        ref = self.collection(parts[0])
        for i in range(1, len(parts)):
            ref = ref.document(parts[i]) if i % 2 else ref.collection(parts[i])
        return ref


def client(*args, **kwargs):
    return Client()
