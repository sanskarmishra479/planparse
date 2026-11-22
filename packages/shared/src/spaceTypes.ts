// Master list of space types. The AI must choose one of these codes;
// anything that does not fit becomes "other" and keeps its original label.
// In Phase 1 this list is seeded into the `space_types` table.

export const SPACE_TYPE_GROUPS = {
  retail: [
    "shop", "showroom", "store", "kiosk", "anchor_store", "hypermarket",
    "department_store", "food_court", "food_stall", "restaurant", "cafe",
    "cinema_screen", "gaming_zone", "bank", "atm",
  ],
  commercial_residential: [
    "office", "cabin", "conference_room", "coworking", "apartment",
    "bedroom", "living_room", "dining_room", "kitchen", "balcony",
    "family_room", "pantry", "laundry", "closet", "garage", "patio",
  ],
  circulation: [
    "lobby", "atrium", "foyer", "passage", "otla", "entrance",
    "emergency_exit", "staircase", "lift", "fire_lift", "escalator",
    "travelator", "ramp", "porch", "portico",
  ],
  sanitary: [
    "toilet_ladies", "toilet_gents", "toilet_accessible", "toilet",
    "wc", "urinal", "bath", "washbasin_area", "baby_care",
  ],
  services: [
    "electrical_room", "dg_room", "transformer", "hvac_room", "pump_room",
    "fire_control_room", "shaft", "fire_duct", "lift_machine_room",
    "water_tank", "stp", "meter_room", "garbage_room", "server_room",
  ],
  back_of_house: [
    "storage", "loading_dock", "service_corridor", "security_room",
    "admin_office", "staff_room", "management_office",
  ],
  parking_outdoor: [
    "parking_bay", "driveway", "basement_ramp", "drop_off", "road",
    "landscape", "open_space", "setback",
  ],
  sub_space: ["loft", "mezzanine", "cubicle"],
  other: ["other"],
} as const;

export type SpaceTypeGroup = keyof typeof SPACE_TYPE_GROUPS;
export type SpaceType = (typeof SPACE_TYPE_GROUPS)[SpaceTypeGroup][number];

export const SPACE_TYPES = Object.values(SPACE_TYPE_GROUPS).flat() as SpaceType[];

/** Types that are normally rented/sold. */
export const LEASABLE_TYPES = new Set<SpaceType>([
  ...SPACE_TYPE_GROUPS.retail,
  "office", "cabin", "coworking", "apartment",
]);
