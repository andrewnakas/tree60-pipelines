"""
Drop-in replacement for the slice of `google.cloud.storage` the Tree 60 raster
pipelines use, backed by Cloudflare R2 over its S3-compatible API.

The pipeline modules were written against GCS. Rather than rewriting their logic
(and risking a change in the imagery they produce), they keep calling
`storage.Client()` / `bucket.blob(...)` / `blob.upload_from_string(...)` and this
module answers those calls against R2. A module is ported by changing one import:

    from google.cloud import storage        ->  import r2_storage as storage

Objects are public through the tree60-api Worker rather than a bucket ACL, so
`make_public()` is a no-op and `public_url` points at that route.
"""

import io
import logging
import os

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
BUCKET = os.environ.get("R2_BUCKET", "tree60-data")
# Objects are served by the Worker route rather than a public bucket domain, so
# everything stays on the site's own origin (which the native apps hard-code).
PUBLIC_BASE = os.environ.get("R2_PUBLIC_BASE", "https://tree60weather.com/api/data")


def _client():
    key = os.environ.get("R2_ACCESS_KEY_ID")
    secret = os.environ.get("R2_SECRET_ACCESS_KEY")
    if not (key and secret and ACCOUNT_ID):
        raise RuntimeError(
            "R2 credentials missing: set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY"
        )
    return boto3.client(
        "s3",
        endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
        region_name="auto",
    )


class Blob:
    def __init__(self, name, bucket):
        self.name = name
        self.bucket = bucket
        # Written by callers before upload, mirroring the GCS API.
        self.cache_control = None
        self.content_encoding = None
        self.content_type = None
        self.size = None
        self.updated = None

    @property
    def public_url(self):
        return f"{PUBLIC_BASE}/{self.name}"

    def _extra(self, content_type=None):
        extra = {}
        ct = content_type or self.content_type
        if ct:
            extra["ContentType"] = ct
        if self.cache_control:
            extra["CacheControl"] = self.cache_control
        if self.content_encoding:
            extra["ContentEncoding"] = self.content_encoding
        return extra

    def upload_from_string(self, data, content_type=None):
        if isinstance(data, str):
            data = data.encode("utf-8")
        self.bucket.client.put_object(
            Bucket=self.bucket.name, Key=self.name, Body=data, **self._extra(content_type)
        )
        self.size = len(data)

    def upload_from_file(self, fileobj, content_type=None, rewind=False):
        if rewind:
            fileobj.seek(0)
        data = fileobj.read()
        self.upload_from_string(data, content_type=content_type)

    def upload_from_filename(self, filename, content_type=None):
        with open(filename, "rb") as fh:
            self.upload_from_string(fh.read(), content_type=content_type)

    def download_as_bytes(self):
        obj = self.bucket.client.get_object(Bucket=self.bucket.name, Key=self.name)
        return obj["Body"].read()

    def download_as_text(self, encoding="utf-8"):
        return self.download_as_bytes().decode(encoding)

    def download_to_file(self, fileobj):
        fileobj.write(self.download_as_bytes())

    def exists(self):
        try:
            self.bucket.client.head_object(Bucket=self.bucket.name, Key=self.name)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def reload(self):
        head = self.bucket.client.head_object(Bucket=self.bucket.name, Key=self.name)
        self.size = head.get("ContentLength")
        self.updated = head.get("LastModified")
        self.content_type = head.get("ContentType")
        return self

    def delete(self):
        self.bucket.client.delete_object(Bucket=self.bucket.name, Key=self.name)

    def make_public(self):
        """No-op: the Worker route serves these objects, not a bucket ACL."""
        return None


class Bucket:
    def __init__(self, name, client):
        self.name = name
        self.client = client

    def blob(self, name):
        return Blob(name, self)

    def list_blobs(self, prefix=None, max_results=None):
        paginator = self.client.get_paginator("list_objects_v2")
        kwargs = {"Bucket": self.name}
        if prefix:
            kwargs["Prefix"] = prefix
        out = []
        for page in paginator.paginate(**kwargs):
            for item in page.get("Contents", []):
                b = Blob(item["Key"], self)
                b.size = item.get("Size")
                b.updated = item.get("LastModified")
                out.append(b)
                if max_results and len(out) >= max_results:
                    return out
        return out

    def copy_blob(self, blob, destination_bucket, new_name=None):
        dest = new_name or blob.name
        self.client.copy_object(
            Bucket=destination_bucket.name,
            Key=dest,
            CopySource={"Bucket": self.name, "Key": blob.name},
        )
        return Blob(dest, destination_bucket)


_warned_names = set()


def _resolve(name):
    """
    Ported modules still pass the old GCS bucket name. There is exactly one R2
    bucket, so any legacy name resolves to it rather than 400ing on a bucket that
    does not exist here.
    """
    if name and name != BUCKET and name not in _warned_names:
        _warned_names.add(name)
        logger.info("mapping legacy bucket %r -> R2 bucket %r", name, BUCKET)
    return BUCKET


class Client:
    def __init__(self, *args, **kwargs):
        self._s3 = _client()

    def bucket(self, name=None):
        return Bucket(_resolve(name), self._s3)

    def get_bucket(self, name=None):
        return self.bucket(name)

    def list_blobs(self, bucket_or_name, prefix=None):
        name = bucket_or_name if isinstance(bucket_or_name, str) else bucket_or_name.name
        return self.bucket(name).list_blobs(prefix=prefix)


def bucket(name=None):
    """Mirrors firebase_admin.storage.bucket() for modules that use that form."""
    return Client().bucket(name)
