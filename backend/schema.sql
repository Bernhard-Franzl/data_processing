CREATE TABLE `rooms` (
  `id` integer PRIMARY KEY,
  `name` text NOT NULL,
  `building_name` text NOT NULL,
  `number_doors` integer DEFAULT 1,
  `capacity` integer NOT NULL
);

CREATE TABLE `doors` (
  `id` integer PRIMARY KEY,
  `room_id` integer
);

CREATE TABLE `events` (
  `id` integer PRIMARY KEY,
  `door_id` integer,
  `created_at` timestamp,
  `entering` boolean,
  `in_support_count` integer,
  `out_support_count` integer,
  `sensor_one_support_count` integer,
  `sensor_two_support_count` integer
);

CREATE TABLE `course_dates` (
  `id` integer PRIMARY KEY,
  `course_id` integer NOT NULL,
  `start_time` timestamp NOT NULL,
  `end_time` timestamp NOT NULL,
  `room_id` integer NOT NULL,
  `remark` text,
  `participants` integer
);

CREATE TABLE `courses` (
  `id` integer PRIMARY KEY,
  `number` text NOT NULL,
  `name` text NOT NULL,
  `type` text NOT NULL,
  `capacity` integer,
  `registrations` integer,
  `study_program` text,
  `faculty` text,
  `institute` text
);

ALTER TABLE `doors` ADD FOREIGN KEY (`room_id`) REFERENCES `rooms` (`id`);

ALTER TABLE `events` ADD FOREIGN KEY (`door_id`) REFERENCES `doors` (`id`);

ALTER TABLE `course_dates` ADD FOREIGN KEY (`course_id`) REFERENCES `courses` (`id`);

ALTER TABLE `course_dates` ADD FOREIGN KEY (`room_id`) REFERENCES `rooms` (`id`);
