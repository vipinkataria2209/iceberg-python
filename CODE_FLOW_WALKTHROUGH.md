# PyIceberg REST Catalog - Code Flow Walkthrough

## 1. LOADING THE CATALOG

### Step 1a: User Code
```python
from pyiceberg.catalog import load_catalog

catalog = load_catalog(
    "rest",
    **{
        "type": "rest",
        "uri": "http://localhost:8181",
        "warehouse": "s3://my-bucket/warehouse",
    }
)
```

### Step 1b: Inside `load_catalog()` - File: `pyiceberg/catalog/__init__.py:221`
```python
def load_catalog(name: str | None = None, **properties: str | None) -> Catalog:
    """Load the catalog based on the properties."""
    
    # 1. Get config from environment/config files
    env = _ENV_CONFIG.get_catalog_config(name)
    conf: RecursiveDict = merge_config(env or {}, cast(RecursiveDict, properties))
    
    # 2. Determine catalog type (REST, HIVE, GLUE, etc.)
    catalog_type = CatalogType(provided_catalog_type.lower())  # "rest"
    
    # 3. Return appropriate catalog instance
    return AVAILABLE_CATALOGS[catalog_type](name, cast(dict[str, str], conf))
```

### Step 1c: Loading REST Catalog - File: `pyiceberg/catalog/__init__.py:130`
```python
def load_rest(name: str, conf: Properties) -> Catalog:
    from pyiceberg.catalog.rest import RestCatalog
    return RestCatalog(name, **conf)
```

### Step 1d: RestCatalog Initialization - File: `pyiceberg/catalog/rest/__init__.py:200+`
```python
class RestCatalog(Catalog):
    def __init__(self, name: str, **properties: str):
        super().__init__(name, **properties)
        # Creates HTTP session with auth, SSL, etc.
        self._session: Session = self._create_session()
```

**Result**: A `RestCatalog` object with an HTTP session ready to make REST API calls

---

## 2. CREATING A NAMESPACE

### User Code
```python
catalog.create_namespace("mynamespace")
```

### Code Path - File: `pyiceberg/catalog/rest/__init__.py:751`
```python
@retry(**_RETRY_ARGS)  # Auto-retry on transient failures
def create_namespace(self, namespace: str | Identifier, 
                     properties: Properties = EMPTY_DICT) -> None:
    # 1. Validate and normalize namespace identifier
    namespace_tuple = self._check_valid_namespace_identifier(namespace)
    # Result: ("mynamespace",)
    
    # 2. Build REST request payload
    payload = {
        "namespace": namespace_tuple,  # ("mynamespace",)
        "properties": properties        # {} or custom props
    }
    
    # 3. Send HTTP POST request
    response = self._session.post(
        self.url(Endpoints.create_namespace),  # "/v1/namespaces"
        json=payload
    )
    
    # 4. Handle response & errors
    try:
        response.raise_for_status()  # Raise if status != 2xx
    except HTTPError as exc:
        # Convert 409 → NamespaceAlreadyExistsError
        _handle_non_200_response(exc, {409: NamespaceAlreadyExistsError})
```

### REST API Call Details
- **Endpoint**: `POST http://localhost:8181/v1/namespaces`
- **Request Body**:
  ```json
  {
    "namespace": ["mynamespace"],
    "properties": {}
  }
  ```
- **Response** (on success):
  ```json
  {
    "namespace": ["mynamespace"],
    "properties": {}
  }
  ```

**What Happens at Server**: 
- REST server validates namespace
- Stores namespace metadata in catalog database (could be Nessie, PostgreSQL, etc.)
- Returns metadata to client
- **NO files created yet** - purely metadata

---

## 3. CREATING A TABLE

### User Code
```python
import pyarrow as pa

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("name", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), 128)),
])

table = catalog.create_table("mynamespace.mytable", schema=schema)
```

### Code Path - File: `pyiceberg/catalog/rest/__init__.py:538`
```python
def create_table(
    self,
    identifier: str | Identifier,              # "mynamespace.mytable"
    schema: Union[Schema, "pa.Schema"],        # PyArrow schema
    location: str | None = None,               # Optional: custom location
    partition_spec: PartitionSpec = UNPARTITIONED_PARTITION_SPEC,
    sort_order: SortOrder = UNSORTED_SORT_ORDER,
    properties: Properties = EMPTY_DICT,
) -> Table:
    # 1. Internal method to build & send request
    table_response = self._create_table(
        identifier=identifier,
        schema=schema,
        location=location,
        partition_spec=partition_spec,
        sort_order=sort_order,
        properties=properties,
        stage_create=False,  # Direct create (not staged)
    )
    
    # 2. Convert REST response to Table object
    return self._response_to_table(
        self.identifier_to_tuple(identifier),  # ("mynamespace", "mytable")
        table_response
    )
```

### Detailed: `_create_table()` - File: `pyiceberg/catalog/rest/__init__.py:496`
```python
def _create_table(
    self,
    identifier: str | Identifier,
    schema: Union[Schema, "pa.Schema"],
    location: str | None = None,
    partition_spec: PartitionSpec = UNPARTITIONED_PARTITION_SPEC,
    sort_order: SortOrder = UNSORTED_SORT_ORDER,
    properties: Properties = EMPTY_DICT,
    stage_create: bool = False,
) -> TableResponse:
    # 1. Convert PyArrow schema to Iceberg schema if needed
    iceberg_schema = self._convert_schema_if_needed(
        schema,
        int(properties.get(
            TableProperties.FORMAT_VERSION,
            TableProperties.DEFAULT_FORMAT_VERSION
        ))
    )
    
    # 2. Assign fresh IDs to schema fields (prevents ID conflicts)
    fresh_schema = assign_fresh_schema_ids(iceberg_schema)
    
    # 3. Assign fresh IDs to partition spec
    fresh_partition_spec = assign_fresh_partition_spec_ids(
        partition_spec,
        iceberg_schema,
        fresh_schema
    )
    
    # 4. Assign fresh IDs to sort order
    fresh_sort_order = assign_fresh_sort_order_ids(
        sort_order,
        iceberg_schema,
        fresh_schema
    )
    
    # 5. Parse namespace and table name from identifier
    namespace_and_table = self._split_identifier_for_path(identifier)
    # Result: {"namespace": "mynamespace", "table": "mytable"}
    
    # 6. Clean up location (remove trailing slash)
    if location:
        location = location.rstrip("/")
    
    # 7. Build table creation request
    request = CreateTableRequest(
        name=namespace_and_table["table"],        # "mytable"
        location=location,                         # None or custom path
        table_schema=fresh_schema,                 # Iceberg schema with IDs
        partition_spec=fresh_partition_spec,       # Partition config
        write_order=fresh_sort_order,              # Sort order
        stage_create=stage_create,                 # False for direct create
        properties=properties,                     # Table properties
    )
    
    # 8. Serialize to JSON
    serialized_json = request.model_dump_json().encode(UTF8)
    
    # 9. Send HTTP POST to REST server
    response = self._session.post(
        self.url(
            Endpoints.create_table,                # "/v1/namespaces/{namespace}/tables"
            namespace=namespace_and_table["namespace"]  # "mynamespace"
        ),
        data=serialized_json,
    )
    
    # 10. Handle response
    try:
        response.raise_for_status()
    except HTTPError as exc:
        _handle_non_200_response(exc, {409: TableAlreadyExistsError, ...})
    
    # 11. Parse response to TableResponse object
    return response.json()  # Contains metadata_location, metadata, config
```

### REST API Call Details
- **Endpoint**: `POST http://localhost:8181/v1/namespaces/mynamespace/tables`
- **Request Body** (simplified):
  ```json
  {
    "name": "mytable",
    "location": null,
    "schema": {
      "type": "struct",
      "fields": [
        {"id": 1, "name": "id", "type": "long"},
        {"id": 2, "name": "name", "type": "string"},
        {"id": 3, "name": "embedding", "type": "list", ...}
      ]
    },
    "partition_spec": {"spec_id": 0, "fields": []},
    "write_order": {"order_id": 0, "fields": []},
    "properties": {}
  }
  ```
- **Response** (simplified):
  ```json
  {
    "metadata_location": "s3://my-bucket/warehouse/mynamespace/mytable/metadata/v1.metadata.json",
    "metadata": {
      "format_version": 2,
      "table_uuid": "...",
      "location": "s3://my-bucket/warehouse/mynamespace/mytable",
      "last_sequence_number": 0,
      "properties": {},
      "current_snapshot_id": null,
      "snapshots": []
    },
    "config": {...}
  }
  ```

### Converting Response to Table Object - File: `pyiceberg/catalog/rest/__init__.py:458`
```python
def _response_to_table(
    self,
    identifier_tuple: tuple[str, ...],           # ("mynamespace", "mytable")
    table_response: TableResponse
) -> Table:
    return Table(
        identifier=identifier_tuple,
        metadata_location=table_response.metadata_location,
        # s3://my-bucket/warehouse/mynamespace/mytable/metadata/v1.metadata.json
        
        metadata=table_response.metadata,
        # TableMetadata object with schema, snapshots, etc.
        
        io=self._load_file_io(
            {**table_response.metadata.properties, **table_response.config},
            table_response.metadata_location
        ),
        # FileIO object for S3 (or other configured storage)
        
        catalog=self,
        # Reference back to RestCatalog
        
        config=table_response.config,
        # Configuration (endpoint URL, auth, etc.)
    )
```

**Result**: A `Table` object with:
- Empty metadata (no snapshots, no data files yet)
- Reference to catalog
- FileIO for storage access
- Ready for data operations (append, scan, etc.)

---

## 4. APPENDING DATA TO TABLE

### User Code
```python
import pyarrow as pa

data = pa.Table.from_pydict({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "embedding": [vec1, vec2, vec3],  # Lists of 128 floats
})

table.append(data)
```

### Code Path - File: `pyiceberg/table/__init__.py:477`
```python
def append(
    self,
    df: pa.Table,                              # PyArrow table with data
    snapshot_properties: dict[str, str] = EMPTY_DICT,
    branch: str | None = MAIN_BRANCH
) -> None:
    """Shorthand API for appending a PyArrow table."""
    
    # 1. Validate PyArrow table
    if not isinstance(df, pa.Table):
        raise ValueError(f"Expected PyArrow table, got: {df}")
    
    # 2. Check schema compatibility
    from pyiceberg.io.pyarrow import _check_pyarrow_schema_compatible, _dataframe_to_data_files
    
    downcast_ns_timestamp_to_us = Config().get_bool(DOWNCAST_NS_TIMESTAMP_TO_US_ON_WRITE) or False
    
    _check_pyarrow_schema_compatible(
        self.table_metadata.schema(),      # Iceberg schema
        provided_schema=df.schema,         # PyArrow schema
        downcast_ns_timestamp_to_us=downcast_ns_timestamp_to_us,
        format_version=self.table_metadata.format_version,
    )
    
    # 3. Start append snapshot transaction
    with self._append_snapshot_producer(snapshot_properties, branch=branch) as append_files:
        # 4. Skip if dataframe is empty
        if df.shape[0] > 0:
            # 5. Convert dataframe to data files (writes Parquet to storage)
            data_files = list(
                _dataframe_to_data_files(
                    table_metadata=self.table_metadata,
                    write_uuid=append_files.commit_uuid,  # Unique write ID
                    df=df,                                # The actual data
                    io=self._table.io,                    # FileIO for S3
                )
            )
            
            # 6. Register each data file in snapshot
            for data_file in data_files:
                append_files.append_data_file(data_file)
        
        # 7. When exiting 'with' block, snapshot is committed
        # Updates table metadata JSON via REST API
```

### Detailed: `_dataframe_to_data_files()` - File: `pyiceberg/io/pyarrow.py:2812`
```python
def _dataframe_to_data_files(
    table_metadata: TableMetadata,
    df: pa.Table,                    # PyArrow table: [1M rows]
    io: FileIO,                      # S3 FileIO
    write_uuid: uuid.UUID | None = None,
    counter: itertools.count[int] | None = None,
) -> Iterable[DataFile]:
    """Convert a PyArrow table into DataFiles.
    
    Returns an iterable that supplies datafiles that represent the table.
    """
    
    # 1. Set defaults
    counter = counter or itertools.count(0)
    write_uuid = write_uuid or uuid.uuid4()
    # Result: write_uuid = "abc-123-def-456"
    
    # 2. Get target file size from table properties
    target_file_size: int = property_as_int(
        properties=table_metadata.properties,
        property_name=TableProperties.WRITE_TARGET_FILE_SIZE_BYTES,
        default=TableProperties.WRITE_TARGET_FILE_SIZE_BYTES_DEFAULT,  # 128MB
    )
    
    # 3. Get name mapping (for schema)
    name_mapping = table_metadata.schema().name_mapping
    
    # 4. Convert PyArrow schema to Iceberg schema for writing
    task_schema = pyarrow_to_schema(
        df.schema,
        name_mapping=name_mapping,
        downcast_ns_timestamp_to_us=Config().get_bool(DOWNCAST_NS_TIMESTAMP_TO_US_ON_WRITE) or False,
        format_version=table_metadata.format_version,
    )
    
    # 5. Check if table is partitioned
    if table_metadata.spec().is_unpartitioned():
        # UNPARTITIONED: Write to single location
        yield from write_file(
            io=io,
            table_metadata=table_metadata,
            tasks=(
                WriteTask(
                    write_uuid=write_uuid,
                    task_id=next(counter),              # 0, 1, 2, ...
                    record_batches=batches,             # Chunked data
                    schema=task_schema,
                )
                for batches in bin_pack_arrow_table(df, target_file_size)
                # Splits 1M rows into chunks of ~128MB each
            ),
        )
    else:
        # PARTITIONED: Group by partition values, write to partition-specific locations
        partitions = _determine_partitions(
            spec=table_metadata.spec(),
            schema=table_metadata.schema(),
            arrow_table=df
        )
        # Groups rows by partition key values
        
        yield from write_file(
            io=io,
            table_metadata=table_metadata,
            tasks=(
                WriteTask(
                    write_uuid=write_uuid,
                    task_id=next(counter),
                    record_batches=batches,
                    partition_key=partition.partition_key,  # e.g., {year: 2024}
                    schema=task_schema,
                )
                for partition in partitions
                for batches in bin_pack_arrow_table(
                    partition.arrow_table_partition,
                    target_file_size
                )
            ),
        )
```

### Detailed: `write_file()` - The Actual Writing
The `write_file()` function:
1. **Creates Parquet files** in storage (e.g., S3)
   - File path: `s3://my-bucket/warehouse/mynamespace/mytable/data/abc-123-def-456/00000-0.parquet`
   - Contains actual data (1M rows serialized to Parquet)

2. **Collects file statistics**
   - Row count: 1000000
   - File size: 128MB
   - Column statistics (min/max values for pruning)
   - Null counts

3. **Creates DataFile object**
   ```python
   DataFile(
       file_path="s3://my-bucket/warehouse/mynamespace/mytable/data/abc-123-def-456/00000-0.parquet",
       file_format=FileFormat.PARQUET,
       partition={},  # Empty for unpartitioned
       record_count=1000000,
       file_size_in_bytes=134217728,
       column_sizes={0: 50MB, 1: 30MB, 2: 50MB},  # Per-column sizes
       value_counts={0: 1000000, 1: 1000000, 2: 1000000},
       null_value_counts={0: 0, 1: 0, 2: 0},
       column_stats=[...],
       # ... more metadata
   )
   ```
   This is yielded back to the calling function.

### Back to `append()`: Committing the Snapshot
```python
with self._append_snapshot_producer(snapshot_properties, branch=branch) as append_files:
    # ... write data files ...
    for data_file in data_files:
        append_files.append_data_file(data_file)
    # When exiting, __exit__() is called:
```

The context manager's `__exit__()` method (automatic on leaving `with` block):
1. **Creates a new Snapshot** with all the DataFiles
2. **Serializes Snapshot to JSON**
3. **Creates a Manifest** (Avro file) listing all DataFiles
4. **Writes Manifest to storage**: `s3://...../metadata/manifests/abc-manifest.avro`
5. **Updates Table Metadata** with new snapshot pointer
6. **Writes Table Metadata JSON**: `s3://...../metadata/v2.metadata.json`
7. **Commits to catalog** via REST API:
   ```
   PUT /v1/namespaces/mynamespace/tables/mytable
   
   {
     "updates": [
       {
         "action": "set-current-snapshot",
         "snapshot-id": 123456789
       }
     ],
     "requirements": [...]
   }
   ```

---

## 5. READING DATA FROM TABLE

### User Code
```python
result = table.scan().to_arrow()
# Or with filter:
result = table.scan().filter("category = 'A'").to_arrow()
```

### Code Path - File: `pyiceberg/table/__init__.py:435`
```python
def scan(
    self,
    row_filter: BooleanExpression | None = None,
    selected_columns: str | Iterable[str] | None = None,
    case_sensitive: bool = True,
    snapshot_id: int | None = None,
    options: Properties = EMPTY_DICT,
    branch: str | None = MAIN_BRANCH,
) -> TableScan:
    """Plan a scan of the table."""
    
    # 1. Create TableScan object
    return TableScan(
        table=self,
        row_filter=row_filter,
        selected_columns=selected_columns,
        case_sensitive=case_sensitive,
        snapshot_id=snapshot_id,
        options=options,
        branch=branch,
    )

def to_arrow(self) -> pa.Table:
    """Execute scan and return PyArrow table."""
    
    # Calls the scan executor (varies by table type)
    # For standard tables: PyArrowTableReader
    # For vector tables: VectorTableReader
    
    return self.execute()
    # 1. Reads current Snapshot from table metadata
    # 2. Gets Manifest file listing all data files
    # 3. For each DataFile:
    #    - Reads Parquet from S3
    #    - Applies row filter (if any)
    #    - Projects columns (if specified)
    # 4. Concatenates all results
    # 5. Returns as PyArrow table
```

---

## FILE ORGANIZATION IN STORAGE

After creating table and appending data, S3 looks like:

```
s3://my-bucket/warehouse/
├── mynamespace/
│   └── mytable/
│       ├── metadata/
│       │   ├── v1.metadata.json           ← Initial table metadata
│       │   ├── v2.metadata.json           ← After append
│       │   ├── v3.metadata.json           ← After another append
│       │   └── manifests/
│       │       ├── abc-manifest.avro      ← Manifest for snapshot 1
│       │       └── def-manifest.avro      ← Manifest for snapshot 2
│       │
│       └── data/
│           ├── abc-123-def-456/           ← Write UUID
│           │   ├── 00000-0.parquet        ← Data file 1
│           │   ├── 00001-1.parquet        ← Data file 2
│           │   └── ...
│           └── xyz-789-uvw-012/
│               ├── 00000-0.parquet
│               └── ...
```

### File Descriptions

1. **Table Metadata** (v1, v2, v3...): 
   - Lists all snapshots
   - Points to current snapshot
   - Schema definition
   - Table properties
   - Format version

2. **Manifest** (*.avro):
   - Lists all DataFiles in a snapshot
   - File paths, counts, statistics
   - Partition info

3. **Data Files** (*.parquet):
   - Actual rows (Parquet compressed)
   - Immutable (never modified)
   - Old files retained for time-travel

---

## SUMMARY: Request/Response Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ USER CODE                                                           │
├─────────────────────────────────────────────────────────────────────┤
│ catalog.create_table("ns.table", schema)                            │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ REST CATALOG (Python)                                              │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Validate schema, assign IDs                                      │
│ 2. POST http://rest-server:8181/v1/namespaces/ns/tables            │
│ 3. Receive metadata_location, metadata                              │
│ 4. Create Table object                                              │
└────────────────┬──────────────────────────────┬────────────────────┘
                 │ HTTP POST                     │ HTTP Response
                 ▼                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ REST SERVER (e.g., Nessie, Apache Iceberg REST)                   │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Validate request                                                 │
│ 2. Generate table UUID                                              │
│ 3. Create table metadata object                                     │
│ 4. Store in catalog database                                        │
│ 5. Assign metadata location (S3 path)                               │
│ 6. Write initial metadata JSON to S3                                │
│ 7. Return response to client                                        │
└────────────────┬──────────────────────────────┬────────────────────┘
                 │                               │
                 └──────────────────┬────────────┘
                                    │
                                    ▼
                      ┌──────────────────────────────┐
                      │ S3 Storage                   │
                      ├──────────────────────────────┤
                      │ metadata/v1.metadata.json    │
                      │ (empty snapshots array)      │
                      └──────────────────────────────┘

                    TABLE CREATED (NO DATA YET)

┌─────────────────────────────────────────────────────────────────────┐
│ USER CODE: Append Data                                              │
├─────────────────────────────────────────────────────────────────────┤
│ table.append(pyarrow_table)                                         │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TABLE (Python)                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Convert PyArrow schema to Iceberg schema                          │
│ 2. Start append transaction                                         │
│ 3. Call io._dataframe_to_data_files()                               │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FILE IO (Python - PyArrowFileWriter)                               │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Chunk dataframe (bin packing)                                    │
│ 2. For each chunk:                                                  │
│    a. Write Parquet to S3                                           │
│    b. Collect file statistics                                       │
│    c. Create DataFile object                                        │
│ 3. Yield DataFiles                                                  │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ▼
         ┌─────────────┐
         │ S3 Storage  │
         ├─────────────┤
         │ data/       │
         │ ├── 00000.. │  ◄─── Written here
         │ ├── 00001.. │
         │ └── ...     │
         └─────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ TABLE (Python) - Commit Phase                                       │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Create Snapshot with all DataFiles                               │
│ 2. Serialize Snapshot to Manifest (*.avro)                          │
│ 3. Write Manifest to S3                                             │
│ 4. Update table metadata JSON with new snapshot                     │
│ 5. Write metadata JSON to S3                                        │
│ 6. PUT /v1/namespaces/ns/tables/table (commit to catalog)           │
└──────────────┬───────────────────────────────────────────────────────┘
               │ HTTP PUT (with requirements/updates)
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ REST SERVER                                                         │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Validate requirements (concurrency check)                        │
│ 2. Apply updates (set current snapshot)                             │
│ 3. Store new metadata in database                                   │
│ 4. Return success                                                   │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ▼
         ┌──────────────────┐
         │ S3 Storage       │
         ├──────────────────┤
         │ metadata/        │
         │ ├── v1.json      │
         │ ├── v2.json   ◄──── Updated
         │ └── manifests/   │
         │     └── *.avro   │  ◄─── Created
         └──────────────────┘

                    APPEND COMPLETE!
```

---

## KEY CONCEPTS

### 1. **Metadata vs Data**
- **Metadata**: JSON files describing schema, snapshots, data files
  - Stored and managed by REST catalog (could be in Nessie, DB, etc.)
  - Small (kilobytes)
  
- **Data**: Actual rows in Parquet/Avro files
  - Stored in warehouse (S3, GCS, etc.)
  - Can be large (terabytes)

### 2. **Immutability**
- Once a Parquet file is written, it's never modified
- New data always creates new files
- Old snapshots keep references to old files
- Enables time-travel and concurrent reads

### 3. **Transactions**
- REST catalog ensures ACID properties
- Requirements (e.g., current snapshot must be X)
- Updates (e.g., set current snapshot to Y)
- If concurrent write happens, update fails with proper error

### 4. **REST Catalog Benefits**
- Decouples metadata storage from data storage
- Can use any backend for metadata (Nessie, PostgreSQL, etc.)
- Enables multi-table transactions
- Better for cloud deployments

---

## CODE ENTRY POINTS

| What You Do | File | Function |
|---|---|---|
| Load catalog | `pyiceberg/catalog/__init__.py` | `load_catalog()` |
| Create namespace | `pyiceberg/catalog/rest/__init__.py` | `create_namespace()` |
| Create table | `pyiceberg/catalog/rest/__init__.py` | `create_table()` |
| Append data | `pyiceberg/table/__init__.py` | `append()` |
| Scan data | `pyiceberg/table/__init__.py` | `scan()` |
| Write files | `pyiceberg/io/pyarrow.py` | `_dataframe_to_data_files()` |
| Table metadata | `pyiceberg/table/metadata.py` | `TableMetadata` class |

---

## CONTRIBUTIONS YOU COULD MAKE

1. **Add new catalog type** (e.g., DynamoDB for metadata)
   - Implement `Catalog` interface
   - Add loader in `__init__.py`

2. **Improve file I/O** (e.g., better Parquet compression)
   - Modify `_dataframe_to_data_files()`
   - Tune `bin_pack_arrow_table()`

3. **Add vector search** (e.g., FAISS integration)
   - Extend `TableScan` for vector queries
   - Add indexing in snapshot metadata

4. **Optimize REST API calls** (e.g., batch operations)
   - Reduce number of round-trips
   - Add request coalescing

5. **Improve error handling**
   - Better error messages
   - Recovery strategies

Happy contributing! 🚀
