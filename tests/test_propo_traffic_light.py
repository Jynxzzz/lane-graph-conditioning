from waymo_open_dataset.protos import scenario_pb2

proto = scenario_pb2.Scenario()
proto.ParseFromString(proto_string)

print("Has traffic light?", proto.HasField("log_traffic_light"))
print(
    "Field content?",
    (
        proto.log_traffic_light
        if proto.HasField("log_traffic_light")
        else "❌ Not present"
    ),
)
