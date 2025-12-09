import weaviate # type: ignore
from prettytable import PrettyTable    # type: ignore

def main():
    client = weaviate.connect_to_local()
    try:
        print("Status:", client.is_ready())

        COLL = "IrishSIPassages"
        coll = client.collections.get(COLL)

        res = coll.query.fetch_objects(limit=50)

        if not res.objects:
            print("No documents found in collection:", COLL)
            return

        # Prepare a neat table
        table = PrettyTable()
        table.field_names = ["Title", "Doc ID", "Tags", "Preview"]

        for obj in res.objects:
            props = obj.properties
            title = props.get("title", "")
            doc_id = props.get("doc_id", "")
            tags = ", ".join(props.get("tags", []) or [])
            preview = (props.get("content", "")[:60] + "...") if props.get("content") else ""
            table.add_row([title, doc_id, tags, preview])

        print("\n Documents in Collection ")
        print(table)

    finally:
        client.close()

if __name__ == "__main__":
    main()
