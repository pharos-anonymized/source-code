import { Link } from "react-router";

const demoData = `{
  "devices": [
    {
      "uid": "vehicle/10001",
      "position": [24.0, 48.0, 26.0],
      "velocity": [0.0, 0.0, 0.0],
      "ts": 1742393414649,
      "include_area": [22.0, 46.0, 24.0, 26.0, 50.0, 28.0],
      "target_pos": [24.0, 48.0, 26.0]
    },
    // ...
  ],
  "humans": [
    {
      "hid": "human/10001",
      "position": [24.0, 1.0, 26.0],
      "velocity": [1.0, 0.0, 0.0],
      "ts": 1742393414649,
    },
    // ...
  ],
  "buildings": [
    {
      "id": "building/10001",
      "bbox": [22.0, 46.0, 24.0, 26.0, 50.0, 28.0]
    },
    // ...
  ]
}
`;

const classes = {
  number: "text-orange-400",
  key: "text-red-400",
  string: "text-green-400",
  boolean: "text-red-500",
  null: "text-purple-500",
};

const syntaxHighlight = (json: string) => {
  if (!json) return ""; // no JSON from response

  json = json
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  return json.replace(
    /("(\\u[\da-fA-F]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g,
    (match) => {
      let cls = classes.number;

      if (/^"/.test(match)) {
        cls = /:$/.test(match) ? classes.key : classes.string;
      } else if (/^(true|false)$/.test(match)) {
        cls = classes.boolean;
      } else if (match === "null") {
        cls = classes.null;
      }

      return `<span class="${cls}">${match}</span>`;
    }
  );
};

export const HelpPage = () => {
  return (
    <div className="w-full h-full p-8">
      <div className="prose mx-auto prose-code:before:content-none prose-code:after:content-none">
        <h2>Usage Instructions</h2>
        <p>
          After entering the{" "}
          <Link to="/" className="text-primary">
            Home Page
          </Link>
          , you can drag and drop files onto the webpage (
          <a href="/vis-demo.json" target="_blank" className="text-primary">
            sample file
          </a>
          ). The uploaded file should be in JSON format containing device data
          with the following structure:
        </p>
        <pre dangerouslySetInnerHTML={{ __html: syntaxHighlight(demoData) }} />
        <p>
          The data file contains device and personnel data structures as
          follows:
        </p>
        <h3>Device Data</h3>
        <ul>
          <li>
            <code>uid</code> is the device ID
          </li>
          <li>
            <code>position</code> is the device position,
            <code>[x, y, z]</code>, where y is the height
          </li>
          <li>
            <code>velocity</code> is the device velocity,
            <code>[vx, vy, vz]</code>
          </li>
          <li>
            <code>ts</code> is the timestamp, integer type, in milliseconds
          </li>
          <li>
            <code>include_area</code> is the safety space,
            <code>[minX, minY, minZ, maxX, maxY, maxZ]</code>
          </li>
          <li>
            <code>target_pos</code> is the device target position,
            <code>[x, y, z]</code>, where y is the height
          </li>
        </ul>
        <h3>Human Data</h3>
        <ul>
          <li>
            <code>hid</code> is the human ID
          </li>
          <li>
            <code>position</code> is the human position,
            <code>[x, y, z]</code>, where y is the height
          </li>
          <li>
            <code>velocity</code> is the human velocity,
            <code>[vx, vy, vz]</code>
          </li>
          <li>
            <code>ts</code> is the timestamp, integer type, in milliseconds
          </li>
        </ul>
        <h3>Building Data</h3>
        <ul>
          <li>
            <code>id</code> is the building ID
          </li>
          <li>
            <code>bbox</code> is the building bounding box,
            <code>[minX, minY, minZ, maxX, maxY, maxZ]</code>
          </li>
        </ul>
        <p>
          After uploading, you can see 3D models of all devices and adjust the
          timeline to view data at different times.
        </p>
      </div>
    </div>
  );
};
